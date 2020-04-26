import zuds
import sys
import sqlalchemy as sa
import pandas as pd

MAX_IMGS = 50

outfile_name = sys.argv[1]

r = zuds.ReferenceImage

sub = zuds.DBSession().query(zuds.ScienceImage.id, sa.func.rank().over(
    order_by=zuds.ScienceImage.maglimit.desc(),
    partition_by=(zuds.ScienceImage.field,
                  zuds.ScienceImage.ccdid,
                  zuds.ScienceImage.qid,
                  zuds.ScienceImage.fid)).label('rank')).outerjoin(
    r, sa.and_(zuds.ScienceImage.field == r.field,
               zuds.ScienceImage.ccdid == r.ccdid,
               zuds.ScienceImage.qid == r.qid,
               zuds.ScienceImage.fid == r.fid,
               r.version == zuds.REFERENCE_VERSION)).filter(r.id == None,
    zuds.ScienceImage.field.in_(zuds.ACTIVE_FIELDS),
    zuds.ScienceImage.infobits == 0,
    zuds.ScienceImage.seeing > 1.7,
    zuds.ScienceImage.seeing < 2.5,
    zuds.ScienceImage.maglimit < 22,
    zuds.ScienceImage.maglimit > 19.2,
    zuds.ScienceImage.obsdate < '2020-01-01'
).subquery()


result = zuds.DBSession().query(
    sub.c.id, zuds.MaskImage.id
).select_from(sub).join(
    zuds.MaskImage,
    zuds.MaskImage.parent_image_id == sub.c.id
).filter(sub.c.rank < MAX_IMGS)

final = pd.DataFrame(result.all(), columns=['id1', 'id2'])

with open(outfile_name, 'w') as f:
    f.write('\n'.join(final['id1'].tolist()) + '\n')
    f.write('\n'.join(final['id2'].tolist()) + '\n')
