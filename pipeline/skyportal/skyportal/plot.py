import numpy as np
import pandas as pd

from bokeh.core.json_encoder import serialize_json
from bokeh.core.properties import List, String
from bokeh.document import Document
from bokeh.layouts import row, column
from bokeh.models import CustomJS, DatetimeTickFormatter, HoverTool, Range1d, Slider
from bokeh.models.widgets import CheckboxGroup, TextInput, Panel, Tabs
from bokeh.palettes import viridis
from bokeh.plotting import figure, ColumnDataSource
from bokeh.util.compiler import bundle_all_models
from bokeh.util.serialization import make_id

from sncosmo.photdata import PhotometricData
from astropy.table import Table

from skyportal.models import (DBSession, Source, ForcedPhotometry,
                              Instrument, Telescope)


SPEC_LINES = {
    'H': ([3970, 4102, 4341, 4861, 6563], '#ff0000'),
    'He': ([3886, 4472, 5876, 6678, 7065], '#002157'),
    'He II': ([3203, 4686], '#003b99'),
    'C II': ([3919, 4267, 6580, 7234, 9234], '#570199'),
    'C III': ([4650, 5696], '#a30198'),
    'C IV': ([5801], '#ff0073'),
    'O': ([7772, 7774, 7775, 8447, 9266], '#007236'),
    'O II': ([3727], '#00a64d'),
    'O III': ([4959, 5007], '#00bf59'),
    'Na': ([5890, 5896, 8183, 8195], '#aba000'),
    'Mg': ([2780, 2852, 3829, 3832, 3838, 4571, 5167, 5173, 5184], '#8c6239'),
    'Mg II': ([2791, 2796, 2803, 4481], '#bf874e'),
    'Si II': ([3856, 5041, 5056, 5670, 6347, 6371], '#5674b9'),
    'S II': ([5433, 5454, 5606, 5640, 5647, 6715], '#a38409'),
    'Ca II': ([3934, 3969, 7292, 7324, 8498, 8542, 8662], '#005050'),
    'Fe II': ([5018, 5169], '#f26c4f'),
    'Fe III': ([4397, 4421, 4432, 5129, 5158], '#f9917b')
}
# TODO add groups
# Galaxy lines
#
# 'H': '4341, 4861, 6563;
# 'N II': '6548, 6583;
# 'O I': '6300;'
# 'O II': '3727;
# 'O III': '4959, 5007;
# 'Mg II': '2798;
# 'S II': '6717, 6731'
# 'H': '3970, 4102, 4341, 4861, 6563'
# 'Na': '5890, 5896, 8183, 8195'
# 'He': '3886, 4472, 5876, 6678, 7065'
# 'Mg': '2780, 2852, 3829, 3832, 3838, 4571, 5167, 5173, 5184'
# 'He II': '3203, 4686'
# 'Mg II': '2791, 2796, 2803, 4481'
# 'O': '7772, 7774, 7775, 8447, 9266'
# 'Si II': '3856, 5041, 5056, 5670 6347, 6371'
# 'O II': '3727'
# 'Ca II': '3934, 3969, 7292, 7324, 8498, 8542, 8662'
# 'O III': '4959, 5007'
# 'Fe II': '5018, 5169'
# 'S II': '5433, 5454, 5606, 5640, 5647, 6715'
# 'Fe III': '4397, 4421, 4432, 5129, 5158'
#
# Other
#
# 'Tel: 6867-6884, 7594-7621'
# 'Tel': '#b7b7b7',
# 'H: 4341, 4861, 6563;
# 'N II': 6548, 6583;
# 'O I': 6300;
# 'O II': 3727;
# 'O III': 4959, 5007;
# 'Mg II': 2798;
# 'S II': 6717, 6731'


class CheckboxWithLegendGroup(CheckboxGroup):
    colors = List(String, help="List of legend colors")
    __implementation__ = """
import {empty, input, label, div} from "core/dom"
import * as p from "core/properties"

import {CheckboxGroup, CheckboxGroupView} from "models/widgets/checkbox_group"

export class CheckboxWithLegendGroupView extends CheckboxGroupView
  render: () ->
    super()
    empty(@el)

    active = @model.active
    colors = @model.colors
    for text, i in @model.labels
      inputEl = input({type: "checkbox", value: "#{i}"})
      inputEl.addEventListener("change", () => @change_input())

      if @model.disabled then inputEl.disabled = true
      if i in active then inputEl.checked = true
      attrs = {
        style: "border-left: 12px solid #{colors[i]}; padding-left: 0.3em;"
      }
      labelEl = label(attrs, inputEl, text)
      if @model.inline
        labelEl.classList.add("bk-bs-checkbox-inline")
        @el.appendChild(labelEl)
      else
        divEl = div({class: "bk-bs-checkbox"}, labelEl)
        @el.appendChild(divEl)

    return @

export class CheckboxWithLegendGroup extends CheckboxGroup
  type: "CheckboxWithLegendGroup"
  default_view: CheckboxWithLegendGroupView

  @define {
    colors:   [ p.Array, []    ]
  }
"""


# TODO replace with (script, div) method
def _plot_to_json(plot):
    """Convert plot to JSON objects necessary for rendering with `bokehJS`.

    Parameters
    ----------
    plot : bokeh.plotting.figure.Figure
        Bokeh plot object to be rendered.

    Returns
    -------
    (str, str)
        Returns (docs_json, render_items) json for the desired plot.
    """
    render_items = [{'docid': plot._id, 'elementid': make_id()}]

    doc = Document()
    doc.add_root(plot)
    docs_json_inner = doc.to_json()
    docs_json = {render_items[0]['docid']: docs_json_inner}

    docs_json = serialize_json(docs_json)
    render_items = serialize_json(render_items)
    custom_model_js = bundle_all_models()

    return docs_json, render_items, custom_model_js



# TODO make async so that thread isn't blocked
def photometry_plot(source_id):
    """Create scatter plot of photometry for source.

    Parameters
    ----------
    source_id : int
        ID of source to be plotted.

    Returns
    -------
    (str, str)
        Returns (docs_json, render_items) json for the desired plot.
    """
    color_map = {'ipr': 'yellow', 'rpr': 'red', 'g': 'green'}

    qobj = DBSession().query(ForcedPhotometry, Telescope.nickname.label('telescope')) \
        .join(Instrument).join(Telescope) \
        .filter(ForcedPhotometry.source_id == source_id)
    data = pd.read_sql(qobj.statement, DBSession().bind)

    if data.empty:
        return None, None, None

    objects = qobj.all()
    news = []; subs = []
    for object in objects:
        object = object[0]
        thumbs = object.forcethumbs
        for thumb in thumbs:
            if thumb.type == 'sub':
                subs.append(thumb.file_uri)
            elif thumb.type == 'new':
                news.append(thumb.file_uri)

    data['color'] = [color_map.get(f, 'black') for f in data['filter']]
    data['label'] = [f'{t} {f}-band'
                     for t, f in zip(data['telescope'], data['filter'])]

    data['new_img'] = news
    data['sub_img'] = subs

    data['filter'] = ['ztf' + f for f in data['filter']]


    # normalize everything to a common zeropoint
    photdata = PhotometricData(Table.from_pandas(data[['mjd', 'filter', 'flux', 'fluxerr', 'zp', 'zpsys', 'lim_mag']]))
    normalized = photdata.normalized(zp=25., zpsys='ab')

    data['flux'] = normalized.flux
    data['fluxerr'] = normalized.fluxerr
    data['alpha'] = 1.

    # calculate the magnitudes
    obsind = data['flux'] / data['fluxerr'] >= 5
    data.ix[~obsind, 'mag'] = None
    data.ix[obsind, 'mag'] = -2.5 * np.log10(data[obsind]['flux']) + 25.
    data.ix[~obsind, 'magerr'] = None
    data.ix[obsind, 'magerr'] = np.abs(-2.5 * data[obsind]['fluxerr'] / data[obsind]['flux'] / np.log(10))
    data['obs'] = obsind

    split = data.groupby('label', sort=False)

    plot = figure(
        plot_width=600,
        plot_height=300,
        active_drag='box_zoom',
        tools='box_zoom,wheel_zoom,pan,reset',
        y_range=(np.nanmin(data['flux'] - data['fluxerr']) -
                 np.abs(np.nanmin(data['flux'] - data['fluxerr'])) * 0.1,
                 np.nanmax(data['flux'] + data['fluxerr']) * 1.1)
    )

    imhover = HoverTool(tooltips=[('mjd', '@mjd'), ('flux', '@flux'),
                                  ('filter', '@filter'),
                                  ('fluxerr', '@fluxerr'),
                                  ('mag', '@mag'),
                                  ('magerr', '@magerr'),
                                  ('lim_mag', '@lim_mag'),
                                  ('new_img', '<img src=../@new_img{safe} height=61 width=61 style="float: left; margin: 0px 15px 15px 0px;" border="2">'),
                                  ('sub_img', '<img src=../@sub_img{safe} height=61 width=61 style="float: left; margin: 0px 15px 15px 0px;" border="2">')])
    plot.add_tools(imhover)

    # simpler tool tip for coadded light curve points
    simplehover = HoverTool(tooltips=[('mjd', '@mjd'), ('flux', '@flux'),
                                      ('filter', '@filter'),
                                      ('fluxerr', '@fluxerr'),
                                      ('mag', '@mag'),
                                      ('magerr', '@magerr'),
                                      ('lim_mag', '@lim_mag')])

    plot.add_tools(simplehover)


    model_dict = {}
    for i, (label, df) in enumerate(split):
        key = f'obs{i}'
        model_dict[key] = plot.scatter(
            x='mjd', y='flux',
            color='color',
            marker='circle',
            fill_color='color',
            alpha='alpha',
            source=ColumnDataSource(df)
        )

        imhover.renderers.append(model_dict[key])

        key = f'bin{i}'
        model_dict[key] = plot.scatter(
            x='mjd', y='flux',
            color='color',
            marker='circle',
            fill_color='color',
            source=ColumnDataSource(data=dict(mjd=[], flux=[], fluxerr=[], filter=[], color=[]))
        )

        simplehover.renderers.append(model_dict[key])

        key = 'obserr' + str(i)
        y_err_x = []
        y_err_y = []

        for d, ro in df.iterrows():
            px = ro['mjd']
            py = ro['flux']
            err = ro['fluxerr']

            y_err_x.append((px, px))
            y_err_y.append((py - err, py + err))

        model_dict[key] = plot.multi_line(xs='xs', ys='ys', color='color', alpha='alpha',
                                          source=ColumnDataSource(data=dict(xs=y_err_x, ys=y_err_y,
                                                                  color=df['color'],
                                                                  alpha=[1.] * len(df))))

        key = f'binerr{i}'
        model_dict[key] = plot.multi_line(xs='xs', ys='ys', color='color',
                                          source=ColumnDataSource(data=dict(xs=[], ys=[], color=[])))

    plot.xaxis.axis_label = 'MJD'
    plot.yaxis.axis_label = 'Flux (AB Zeropoint = 25.)'
    plot.toolbar.logo = None

    toggle = CheckboxWithLegendGroup(
        labels=list(data.label.unique()),
        active=list(range(len(data.label.unique()))),
        colors=list(data.color.unique()))

    # TODO replace `eval` with Namespaces
    # https://github.com/bokeh/bokeh/pull/6340
    toggle.callback = CustomJS(args={'toggle': toggle, **model_dict},
                               code="""
        for (let i = 0; i < toggle.labels.length; i++) {
            eval("obs" + i).visible = (toggle.active.includes(i));
            eval("obserr" + i).visible = (toggle.active.includes(i));
            eval("bin" + i).visible = (toggle.active.includes(i));
            eval("binerr" + i).visible = (toggle.active.includes(i));
        }
    """)

    slider = Slider(start=0., end=15., value=0., step=1., title='binsize (days)')

    callback = CustomJS(args={'slider': slider, 'toggle': toggle, **model_dict}, code="""

         var binsize = slider.value;
         var fluxalph = ((binsize == 0) ? 1. : 0.1);

         for (var i = 0; i < toggle.labels.length; i++) {

             var fluxsource = eval("obs" + i).data_source;
             var binsource = eval("bin" + i).data_source;

             var fluxerrsource = eval("obserr" + i).data_source;
             var binerrsource = eval("binerr" + i).data_source;

             var minmjd = Math.min.apply(Math, fluxsource.data['mjd']);

             var date = new Date();     // a new date
             var time = date.getTime(); // the timestamp, not neccessarely using UTC as current time
             var maxmjd = ((time / 86400000) - (date.getTimezoneOffset()/1440) + 40587.);

             binsource.data['mjd'] = [];
             binsource.data['flux'] = [];
             binsource.data['fluxerr'] = [];
             binsource.data['filter'] = [];
             binsource.data['color'] = [];
             binsource.data['lim_mag'] = [];
             binsource.data['mag'] = [];
             binsource.data['magerr'] = [];

             binerrsource.data['xs'] = [];
             binerrsource.data['ys'] = [];
             binerrsource.data['color'] = [];

             for (var j = 0; j < fluxsource.get_length(); j++){
                 fluxsource.data['alpha'][j] = fluxalph;
                 fluxerrsource.data['alpha'][j] = fluxalph;
             }

             if (binsize > 0){ 

                 // now do the binning
                 var k = 0;
                 var curmjd = minmjd;
                 var mjdbins = [curmjd];

                 while (curmjd < maxmjd){
                     curmjd += binsize;
                     mjdbins.push(curmjd);               
                 }

                 var nbins = mjdbins.length - 1;
                 for (var l = 0; l < nbins; l++) {


                     // calculate the flux, fluxerror, and mjd of the bin
                     var flux = [];
                     var weight = [];
                     var mjd = [];
                     var limmag = [];
                     var ivarsum = 0;

                     for (var m = 0; m < fluxsource.get_length(); m++){
                         if ((fluxsource.data['mjd'][m] < mjdbins[l + 1]) && (fluxsource.data['mjd'][m] >= mjdbins[l])){

                             let fluxvar = fluxsource.data['fluxerr'][m] * fluxsource.data['fluxerr'][m];
                             let ivar = 1 / fluxvar;

                             weight.push(ivar);
                             flux.push(fluxsource.data['flux'][m]);
                             mjd.push(fluxsource.data['mjd'][m]);
                             limmag.push(fluxsource.data['lim_mag']);
                             ivarsum += ivar;
                         }
                     }

                     var myflux = 0;
                     var mymjd = 0;

                     if (weight.length == 0){
                         continue;
                     }

                     for (var n = 0; n < weight.length; n++){
                         myflux += weight[n] * flux[n] / ivarsum;
                         mymjd += weight[n] * mjd[n] / ivarsum;
                     }

                     var myfluxerr = Math.sqrt(1 / ivarsum);


                     if (myflux / myfluxerr > 5.){
                         var mymag = -2.5 * Math.log10(myflux) + 25;
                         var mymagerr = Math.abs(-2.5 * myfluxerr  / myflux / Math.log(10));
                     } else {
                         var mymag = null;
                         var mymagerr = null;
                     }

                     var mymaglim = -2.5 * Math.log10(5 * myfluxerr) + 25;   

                     binsource.data['mjd'].push(mymjd);
                     binsource.data['flux'].push(myflux);
                     binsource.data['fluxerr'].push(myfluxerr);
                     binsource.data['filter'].push(fluxsource.data['filter'][0]);
                     binsource.data['color'].push(fluxsource.data['color'][0]);
                     binsource.data['mag'].push(mymag);
                     binsource.data['magerr'].push(mymagerr);
                     binsource.data['lim_mag'].push(mymaglim);

                     binerrsource.data['xs'].push([mymjd, mymjd]);
                     binerrsource.data['ys'].push([myflux - myfluxerr, myflux + myfluxerr]);
                     binerrsource.data['color'].push(fluxsource.data['color'][0]);

                 }
             }

             fluxsource.change.emit();
             binsource.change.emit();

             fluxerrsource.change.emit();
             binerrsource.change.emit();
         }

    """)
    slider.js_on_change('value', callback)

    layout = row(plot, toggle)
    layout = column(slider, layout)

    p1 = Panel(child=layout, title='flux')

    # now make the mag light curve
    ymax = 1.1 * data['lim_mag']
    ymin = 0.9 * data['lim_mag']

    if len(data['obs']) > 0:
        ymax[data['obs']] = (data['mag'] + data['magerr']) * 1.1
        ymin[data['obs']] = (data['mag'] - data['magerr']) * 0.9

    plot = figure(
        plot_width=600,
        plot_height=300,
        active_drag='box_zoom',
        tools='box_zoom,wheel_zoom,pan,reset',
        y_range=(np.nanmax(ymax), np.nanmin(ymin))
    )

    imhover = HoverTool(tooltips=[('mjd', '@mjd'), ('flux', '@flux'),
                                  ('filter', '@filter'),
                                  ('fluxerr', '@fluxerr'),
                                  ('mag', '@mag'),
                                  ('magerr', '@magerr'),
                                  ('lim_mag', '@lim_mag'),
                                  ('new_img', '<img src=../@new_img{safe} height=61 width=61 style="float: left; margin: 0px 15px 15px 0px;" border="2">'),
                                  ('sub_img', '<img src=../@sub_img{safe} height=61 width=61 style="float: left; margin: 0px 15px 15px 0px;" border="2">')])
    plot.add_tools(imhover)

    # simpler tool tip for coadded light curve points
    simplehover = HoverTool(tooltips=[('mjd', '@mjd'), ('flux', '@flux'),
                                      ('filter', '@filter'),
                                      ('fluxerr', '@fluxerr'),
                                      ('mag', '@mag'),
                                      ('magerr', '@magerr'),
                                      ('lim_mag', '@lim_mag')])


    plot.add_tools(imhover)
    plot.add_tools(simplehover)

    model_dict = {}

    for i, (label, df) in enumerate(split):

        key = f'obs{i}'
        model_dict[key] = plot.scatter(
            x='mjd', y='mag',
            color='color',
            marker='circle',
            fill_color='color',
            alpha='alpha',
            source=ColumnDataSource(df[df['obs']])
        )

        imhover.renderers.append(model_dict[key])

        key = f'unobs{i}'
        model_dict[key] = plot.scatter(
            x='mjd', y='lim_mag',
            color='color',
            marker='inverted_triangle',
            fill_color='color',
            alpha='alpha',
            source=ColumnDataSource(df[~df['obs']])
        )

        imhover.renderers.append(model_dict[key])

        key = f'bin{i}'
        model_dict[key] = plot.scatter(
            x='mjd', y='mag',
            color='color',
            marker='circle',
            fill_color='color',
            source=ColumnDataSource(data=pd.DataFrame(data=None, columns=df.columns))
        )

        simplehover.renderers.append(model_dict[key])

        key = 'obserr' + str(i)
        y_err_x = []
        y_err_y = []

        for d, ro in df[df['obs']].iterrows():
            px = ro['mjd']
            py = ro['mag']
            err = ro['magerr']

            y_err_x.append((px, px))
            y_err_y.append((py - err, py + err))

        model_dict[key] = plot.multi_line(xs='xs', ys='ys', color='color', alpha='alpha',
                                          source=ColumnDataSource(data=dict(xs=y_err_x, ys=y_err_y,
                                                                  color=df[df['obs']]['color'],
                                                                  alpha=[1.] * len(df[df['obs']]))))

        key = f'binerr{i}'
        model_dict[key] = plot.multi_line(xs='xs', ys='ys', color='color',
                                          source=ColumnDataSource(data=dict(xs=[], ys=[], color=[])))

        key = f'unobsbin{i}'
        model_dict[key] = plot.scatter(
            x='mjd', y='lim_mag',
            color='color',
            marker='inverted_triangle',
            fill_color='color',
            source=ColumnDataSource(data=dict(mjd=[], lim_mag=[], color=[]))
        )

        key = f'all{i}'
        model_dict[key] = ColumnDataSource(df)

    plot.xaxis.axis_label = 'MJD'
    plot.yaxis.axis_label = 'mag'
    plot.toolbar.logo = None

    toggle = CheckboxWithLegendGroup(
        labels=list(data.label.unique()),
        active=list(range(len(data.label.unique()))),
        colors=list(data.color.unique()))

    # TODO replace `eval` with Namespaces
    # https://github.com/bokeh/bokeh/pull/6340
    toggle.callback = CustomJS(args={'toggle': toggle, **model_dict},
                               code="""
        for (let i = 0; i < toggle.labels.length; i++) {
            eval("obs" + i).visible = (toggle.active.includes(i));
            eval("obserr" + i).visible = (toggle.active.includes(i));
            eval("bin" + i).visible = (toggle.active.includes(i));
            eval("binerr" + i).visible = (toggle.active.includes(i));
            eval("unobs" + i).visible = (toggle.active.includes(i));
            eval("unobsbin" + i).visible = (toggle.active.includes(i));
        }
    """)

    slider = Slider(start=0., end=15., value=0., step=1., title='binsize (days)')

    callback = CustomJS(args={'slider': slider, 'toggle': toggle, **model_dict}, code="""

         var binsize = slider.value;
         var fluxalph = ((binsize == 0) ? 1. : 0.1);

         for (var i = 0; i < toggle.labels.length; i++) {

             var fluxsource = eval("obs" + i).data_source;
             var binsource = eval("bin" + i).data_source;

             var fluxerrsource = eval("obserr" + i).data_source;
             var binerrsource = eval("binerr" + i).data_source;
             
             var unobssource = eval("unobs" + i).data_source;
             var unobsbinsource = eval("unobsbin" + i).data_source;
             
             var allsource = eval("all" + i);
             
             var minmjd = Math.min.apply(Math, fluxsource.data['mjd']);

             var date = new Date();     // a new date
             var time = date.getTime(); // the timestamp, not neccessarely using UTC as current time
             var maxmjd = ((time / 86400000) - (date.getTimezoneOffset()/1440) + 40587.);

             binsource.data['mjd'] = [];
             binsource.data['flux'] = [];
             binsource.data['fluxerr'] = [];
             binsource.data['filter'] = [];
             binsource.data['color'] = [];
             binsource.data['lim_mag'] = [];
             binsource.data['mag'] = [];
             binsource.data['magerr'] = [];

             binerrsource.data['xs'] = [];
             binerrsource.data['ys'] = [];
             binerrsource.data['color'] = [];
             
             unobsbinsource.data['mjd'] = [];
             unobsbinsource.data['lim_mag'] = [];
             unobsbinsource.data['color'] = [];

             for (var j = 0; j < fluxsource.get_length(); j++){
                 fluxsource.data['alpha'][j] = fluxalph;
                 fluxerrsource.data['alpha'][j] = fluxalph;
             }
             
             for (var j = 0; j < unobssource.get_length(); j++){
                 unobssource.data['alpha'][j] = fluxalph;
             }

             if (binsize > 0){ 

                 // now do the binning
                 var k = 0;
                 var curmjd = minmjd;
                 var mjdbins = [curmjd];

                 while (curmjd < maxmjd){
                     curmjd += binsize;
                     mjdbins.push(curmjd);               
                 }

                 var nbins = mjdbins.length - 1;
                 for (var l = 0; l < nbins; l++) {


                     // calculate the flux, fluxerror, and mjd of the bin
                     var flux = [];
                     var weight = [];
                     var mjd = [];
                     var limmag = [];
                     var ivarsum = 0;

                     for (var m = 0; m < allsource.get_length(); m++){
                         if ((allsource.data['mjd'][m] < mjdbins[l + 1]) && (allsource.data['mjd'][m] >= mjdbins[l])){

                             let fluxvar = allsource.data['fluxerr'][m] * allsource.data['fluxerr'][m];
                             let ivar = 1 / fluxvar;

                             weight.push(ivar);
                             flux.push(allsource.data['flux'][m]);
                             mjd.push(allsource.data['mjd'][m]);
                             limmag.push(allsource.data['lim_mag']);
                             ivarsum += ivar;
                         }
                     }

                     var myflux = 0;
                     var mymjd = 0;

                     if (weight.length == 0){
                         continue;
                     }

                     for (var n = 0; n < weight.length; n++){
                         myflux += weight[n] * flux[n] / ivarsum;
                         mymjd += weight[n] * mjd[n] / ivarsum;
                     }

                     var myfluxerr = Math.sqrt(1 / ivarsum);
                     var obs = myflux / myfluxerr > 5;
                     
                     if (obs){
                         var mymag = -2.5 * Math.log10(myflux) + 25;
                         var mymagerr = Math.abs(-2.5 * myfluxerr  / myflux / Math.log(10));
                         var mysource = binsource;
                         
                         binerrsource.data['xs'].push([mymjd, mymjd]);
                         binerrsource.data['ys'].push([mymag - mymagerr, mymag + mymagerr]);
                         binerrsource.data['color'].push(allsource.data['color'][0]);
                                                   
                     } else {
                         var mymag = null;
                         var mymagerr = null;
                         var mysource = unobsbinsource;

                     }
                     
                     var mymaglim = -2.5 * Math.log10(5 * myfluxerr) + 25;   
                     
                     mysource.data['mjd'].push(mymjd);
                     mysource.data['flux'].push(myflux);
                     mysource.data['fluxerr'].push(myfluxerr);
                     mysource.data['filter'].push(fluxsource.data['filter'][0]);
                     mysource.data['color'].push(fluxsource.data['color'][0]);
                     mysource.data['mag'].push(mymag);
                     mysource.data['magerr'].push(mymagerr);
                     mysource.data['lim_mag'].push(mymaglim);


                 }
             }

             fluxsource.change.emit();
             binsource.change.emit();

             fluxerrsource.change.emit();
             binerrsource.change.emit();
             
             unobssource.change.emit();
             unobsbinsource.change.emit();
         }

    """)

    slider.js_on_change('value', callback)

    layout = row(plot, toggle)
    layout = column(slider, layout)

    p2 = Panel(child=layout, title='mag')

    tabs = Tabs(tabs=[p1, p2])
    return _plot_to_json(tabs)


# TODO make async so that thread isn't blocked
def spectroscopy_plot(source_id):
    """TODO normalization? should this be handled at data ingestion or plot-time?"""
    source = Source.query.get(source_id)
    spectra = Source.query.get(source_id).spectra

    if len(spectra) == 0:
        return None, None, None

    color_map = dict(zip([s.id for s in spectra], viridis(len(spectra))))
    data = pd.concat(
        [pd.DataFrame({'wavelength': s.wavelengths,
                       'flux': s.fluxes, 'id': s.id,
                       'instrument': s.instrument.telescope.nickname})
         for i, s in enumerate(spectra)]
    )
    split = data.groupby('id')
    hover = HoverTool(tooltips=[('wavelength', '$x'), ('flux', '$y'),
                                ('instrument', '@instrument')])
    plot = figure(plot_width=600, plot_height=300, sizing_mode='scale_both',
                  tools='box_zoom,wheel_zoom,pan,reset',
                  active_drag='box_zoom')
    plot.add_tools(hover)
    model_dict = {}
    for i, (key, df) in enumerate(split):
        model_dict['s' + str(i)] = plot.line(x='wavelength', y='flux',
                                             color=color_map[key],
                                             source=ColumnDataSource(df))
    plot.xaxis.axis_label = 'Wavelength (Å)'
    plot.yaxis.axis_label = 'Flux'
    plot.toolbar.logo = None

    # TODO how to choose a good default?
    plot.y_range = Range1d(0, 1.03 * data.flux.max())

    toggle = CheckboxWithLegendGroup(labels=[s.instrument.telescope.nickname
                                             for s in spectra],
                                     active=list(range(len(spectra))),
                                     width=100,
                                     colors=[color_map[k] for k, df in split])
    toggle.callback = CustomJS(args={'toggle': toggle, **model_dict},
                               code="""
          for (let i = 0; i < toggle.labels.length; i++) {
              eval("s" + i).visible = (toggle.active.includes(i))
          }
    """)

    elements = CheckboxWithLegendGroup(
        labels=list(SPEC_LINES.keys()),
        active=[], width=80,
        colors=[c for w, c in SPEC_LINES.values()]
    )
    z = TextInput(value=str(source.red_shift), title="z:")
    v_exp = TextInput(value='0', title="v_exp:")
    for i, (wavelengths, color) in enumerate(SPEC_LINES.values()):
        el_data = pd.DataFrame({'wavelength': wavelengths})
        el_data['x'] = el_data['wavelength'] * (1 + source.red_shift)
        model_dict[f'el{i}'] = plot.segment(x0='x', x1='x',
                                            # TODO change limits
                                            y0=0, y1=1e-13, color=color,
                                            source=ColumnDataSource(el_data))
        model_dict[f'el{i}'].visible = False

    # TODO callback policy: don't require submit for text changes?
    elements.callback = CustomJS(args={'elements': elements, 'z': z,
                                       'v_exp': v_exp, **model_dict},
                                 code="""
          let c = 299792.458; // speed of light in km / s
          for (let i = 0; i < elements.labels.length; i++) {
              let el = eval("el" + i);
              el.visible = (elements.active.includes(i))
              el.data_source.data.x = el.data_source.data.wavelength.map(
                  x_i => (x_i * (1 + parseFloat(z.value)) /
                                (1 + parseFloat(v_exp.value) / c))
              );
              el.data_source.change.emit();
          }
    """)
    z.callback = elements.callback
    v_exp.callback = elements.callback

    layout = row(plot, toggle, elements, column(z, v_exp))
    return _plot_to_json(layout)
