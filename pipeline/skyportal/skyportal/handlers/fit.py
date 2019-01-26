from baselayer.app.handlers.base import BaseHandler
from baselayer.app.access import auth_or_token
from pipeline.skyportal.skyportal import fit

import tornado.web


class FitHandler(BaseHandler):

    @auth_or_token
    def post(self):
        data = self.get_json()

        source_id = data['source_id']
        fittype = data['fit_type']

        fixz = fittype == 'gn17'

        try:
            fitmod, fitres = fit.fit_source(source_id, fix_z_to_nearest_neighbor=fixz)
        except KeyError as e:
            self.error(e)
        else:
            self.success({'fitmod': fitmod.json()})
