import db
from datetime import timedelta
import pandas as pd
import sys



# set the stack window size
STACK_WINDOW = 7.  # days
STACK_INTERVAL = timedelta(days=STACK_WINDOW)

def solve_stacks(stackclass=db.ScienceCoadd, target_class=db.ScienceImage):

    # create the date table
    gs = db.sa.func.generate_series
    timetype = db.sa.DateTime(timezone=False)
    mindate = db.sa.cast('2017-01-03', timetype)
    maxdate = db.sa.cast(db.sa.func.now(), timetype)

    lcol = gs(mindate,
              maxdate - STACK_INTERVAL,
              STACK_INTERVAL).label('left')

    rcol = gs(mindate + STACK_INTERVAL,
              maxdate,
              STACK_INTERVAL).label('right')

    daterange = db.DBSession().query(lcol, rcol).subquery()

    target = db.sa.func.array_agg(target_class.id).label('target')
    stacksize = db.sa.func.array_length(target, 1).label('stacksize')
    stackcond = stacksize >= 2
    jcond = db.sa.and_(db.ScienceImage.obsdate > daterange.c.left,
                       db.ScienceImage.obsdate <= daterange.c.right)

    res = db.DBSession().query(db.ScienceImage.field,
                               db.ScienceImage.ccdid,
                               db.ScienceImage.qid,
                               db.ScienceImage.fid,
                               daterange.c.left, daterange.c.right,
                               target).select_from(
        db.sa.join(db.SingleEpochSubtraction, db.ScienceImage.__table__,
                   db.SingleEpochSubtraction.target_image_id ==
                   db.ScienceImage.id).join(
            db.ReferenceImage.__table__, db.ReferenceImage.__table__.c.id ==
                               db.SingleEpochSubtraction.reference_image_id
        ).join(daterange, jcond)
    ).filter(
        db.ScienceImage.seeing < 4.,
        db.ScienceImage.maglimit > 19.2,
        db.ReferenceImage.version == 'zuds5',
        db.ScienceImage.filefracday > 20200107000000
    ).group_by(
        db.ScienceImage.field,
        db.ScienceImage.ccdid,
        db.ScienceImage.qid,
        db.ScienceImage.fid,
        daterange.c.left, daterange.c.right
    ).having(
        stackcond
    ).order_by(
        stacksize.desc()
    ).subquery()

    excludecond = db.sa.and_(
        stackclass.field == res.c.field,
        stackclass.ccdid == res.c.ccdid,
        stackclass.qid == res.c.qid,
        stackclass.fid == res.c.fid,
        stackclass.binleft == res.c.left,
        stackclass.binright == res.c.right
    )


    # get the coadds where there is no existing coadd already
    final = db.DBSession().query(res).outerjoin(
        stackclass, excludecond
    ).filter(
        stackclass.id == None
    )

    return final


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('outfile')
    parser.add_argument('--sub', action='store_true', default=False, dest='sub')
    args = parser.parse_args()

    outfile = args.outfile
    stackclass = db.StackedSubtraction if args.sub else db.ScienceCoadd
    targetclass = db.SingleEpochSubtraction if args.sub else db.ScienceImage

    final = solve_stacks(stackclass=stackclass, target_class=targetclass)
    result = pd.read_sql(final.statement, db.DBSession().get_bind())
    result.to_csv(outfile, index=False)
