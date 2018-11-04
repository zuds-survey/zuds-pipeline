from flask import *
from flask.ext.sqlalchemy import SQLAlchemy

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'psql://'
db = SQLAlchemy(app)


class Subtraction(db.Model):
    """ Subtractions """
    __tablename__ = 'subtraction'

    id = db.Column(db.Integer, primary_key=True)
    candidate = db.relationship('Candidate', backref='sub')
    filename = db.Column(db.String(80))
    ref_filename = db.Column(db.String(80))
    new_filename = db.Column(db.String(80))
    ujd = db.Column(db.Float)
    date = db.Column(db.String(80))
    ra_c = db.Column(db.Float)
    dec_c = db.Column(db.Float)
    ra_ll = db.Column(db.Float)
    dec_ll = db.Column(db.Float)
    ra_ur = db.Column(db.Float)
    dec_ur = db.Column(db.Float)
    ra_lr = db.Column(db.Float)
    dec_lr = db.Column(db.Float)
    ra_ul = db.Column(db.Float)
    dec_ul = db.Column(db.Float)
    height = db.Column(db.Float)
    width = db.Column(db.Float)
    good_pix_area = db.Column(db.Float)
    ptffield = db.Column(db.Integer)
    ccdid = db.Column(db.Integer)
    filter = db.Column(db.String(80))
    lmt_mg_new = db.Column(db.Float)
    lmt_mg_ref = db.Column(db.Float)
    medsky_new = db.Column(db.Float)
    medsky_ref = db.Column(db.Float)
    seeing_new = db.Column(db.Float)
    seeing_ref = db.Column(db.Float)
    skysig_new = db.Column(db.Float)
    skysig_ref = db.Column(db.Float)
    ub1_zp_new = db.Column(db.Float)
    ub1_zp_ref = db.Column(db.Float)
    il = db.Column(db.Float)
    tl = db.Column(db.Integer)
    iu = db.Column(db.Float)
    tu = db.Column(db.Float)
    r = db.Column(db.Float)
    rss = db.Column(db.Float)
    nsx = db.Column(db.Float)
    nsy = db.Column(db.Float)
    objs_extracted = db.Column(db.Integer)
    objs_saved = db.Column(db.Integer)
    sub_zp = db.Column(db.Float)
    stamps_done = db.Column(db.Boolean)

    def __init__(self, **kwargs):
        for arg in kwargs.keys():
            setattr(self, arg, kwargs[arg])

class Image(db.Model):

    pass


class Star(db.Model):
    """Stars from SDSS and IPAC"""
    __tablename__ = "star"
    id = db.Column(db.Integer, primary_key=True)
    ra = db.Column(db.Float)
    dec = db.Column(db.Float)
    candidate = db.relationship('Candidate', backref='star')


class Candidate(db.Model):
    """Candidates SExtracted from subs."""

    __tablename__ = 'candidate'
    id = db.Column(db.Integer, primary_key=True)
    subtraction_id = db.Column(db.Integer, db.ForeignKey('subtraction.id'))
    star_id = db.Column(db.Integer, db.ForeignKey('star.id'))
    number = db.Column(db.Integer)
    x_sub = db.Column(db.Float)
    y_sub = db.Column(db.Float)
    ra = db.Column(db.Float)
    dec = db.Column(db.Float)
    mag = db.Column(db.Float)
    mag_err = db.Column(db.Float)
    flux = db.Column(db.Float)
    flux_err = db.Column(db.Float)
    f_aper = db.Column(db.Float)
    f_aper_err = db.Column(db.Float)
    background = db.Column(db.Float)
    a_image = db.Column(db.Float)
    b_image = db.Column(db.Float)
    fwhm = db.Column(db.Float)
    flag = db.Column(db.Integer)
    x_ref = db.Column(db.Float)
    y_ref = db.Column(db.Float)
    ra_ref = db.Column(db.Float)
    dec_ref = db.Column(db.Float)
    mag_ref = db.Column(db.Float)
    mag_ref_err = db.Column(db.Float)
    a_ref = db.Column(db.Float)
    b_ref = db.Column(db.Float)
    fwhm_ref = db.Column(db.Float)
    mag_aper = db.Column(db.Float)
    mag_aper_err = db.Column(db.Float)
    flux_aper = db.Column(db.Float)
    flux_aper_err = db.Column(db.Float)
    n2sig3 = db.Column(db.Integer)
    n3sig3 = db.Column(db.Integer)
    n2sig5 = db.Column(db.Integer)
    n3sig5 = db.Column(db.Integer)
    nmask = db.Column(db.Integer)
    sym = db.Column(db.Float)
    xint_ref = db.Column(db.Integer)
    yint_ref = db.Column(db.Integer)
    xint_new = db.Column(db.Integer)
    yint_new = db.Column(db.Integer)
    nn_dist = db.Column(db.Float)
    pos_sub = db.Column(db.String(80))
    candidate_mid = db.Column(db.Integer)
    ml_bogus = db.Column(db.Float)
    ml_suspect = db.Column(db.Float)
    ml_unclear = db.Column(db.Float)
    ml_maybe = db.Column(db.Float)
    ml_real = db.Column(db.Float)
    ml_class = db.Column(db.String(80))
    sn = db.Column(db.Integer, db.ForeignKey('supernova.id'))
    ref_stamp_filename = db.Column(db.String(80))
    new_stamp_filename = db.Column(db.String(80))
    sub_stamp_filename = db.Column(db.String(80))
    is_star = db.Column(db.Boolean)
    is_fake = db.Column(db.Boolean)

    def __init__(self, **kwargs):
        for arg in kwargs.keys():
            setattr(self, arg, kwargs[arg])


db.create_all()
