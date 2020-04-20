import zuds
from datetime import timedelta
import pandas as pd
import sys

import sqlalchemy as sa

outfile = sys.argv[1]

zuds.init_db()

# set the stack window size
STACK_WINDOW = 7.  # days
STACK_INTERVAL = timedelta(days=STACK_WINDOW)

# create the date table
gs = sa.func.generate_series
timetype = sa.DateTime(timezone=False)
mindate = sa.cast('2017-01-03', timetype)
maxdate = sa.cast(sa.func.now(), timetype)

lcol = gs(mindate,
          maxdate - STACK_INTERVAL,
          STACK_INTERVAL).label('left')

rcol = gs(mindate + STACK_INTERVAL,
          maxdate,
          STACK_INTERVAL).label('right')

daterange = zuds.DBSession().query(lcol, rcol).subquery()

target = sa.func.array_agg(zuds.ScienceImage.id).label('target')
stacksize = sa.func.array_length(target, 1).label('stacksize')
stackcond = stacksize >= 2
jcond = sa.and_(zuds.ScienceImage.obsdate > daterange.c.left,
                zuds.ScienceImage.obsdate <= daterange.c.right)

res = zuds.DBSession().query(zuds.ScienceImage.field,
                           zuds.ScienceImage.ccdid,
                           zuds.ScienceImage.qid,
                           zuds.ScienceImage.fid,
                           daterange.c.left, daterange.c.right,
                           target).select_from(
    sa.join(zuds.SingleEpochSubtraction, zuds.ScienceImage.__table__,
            zuds.SingleEpochSubtraction.target_image_id ==
            zuds.ScienceImage.id).join(
        zuds.ReferenceImage.__table__, zuds.ReferenceImage.__table__.c.id ==
                                       zuds.SingleEpochSubtraction.reference_image_id
    ).join(daterange, jcond)
).filter(
    zuds.ScienceImage.seeing < 4.,
    zuds.ScienceImage.maglimit > 19.2,
    zuds.ReferenceImage.version == 'zuds5',
    zuds.ScienceImage.filefracday > 20200107000000
).group_by(
    zuds.ScienceImage.field,
    zuds.ScienceImage.ccdid,
    zuds.ScienceImage.qid,
    zuds.ScienceImage.fid,
    daterange.c.left, daterange.c.right
).having(
    stackcond
).order_by(
    stacksize.desc()
).subquery()

excludecond = sa.and_(
    zuds.ScienceCoadd.field == res.c.field,
    zuds.ScienceCoadd.ccdid == res.c.ccdid,
    zuds.ScienceCoadd.qid == res.c.qid,
    zuds.ScienceCoadd.fid == res.c.fid,
    zuds.ScienceCoadd.binleft == res.c.left,
    zuds.ScienceCoadd.binright == res.c.right
)


# get the coadds where there is no existing coadd already
final = zuds.DBSession().query(res).outerjoin(
    zuds.ScienceCoadd, excludecond
).filter(
    zuds.ScienceCoadd.id == None
)

result = pd.read_sql(final.statement, zuds.DBSession().get_bind())
result.to_csv(outfile, index=False)
