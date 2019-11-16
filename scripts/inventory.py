import os
import db
import subprocess
import tarfile
import uuid
from pathlib import Path
from mpi import get_my_share_of_work


def object_from_filename(fname):

    if 'sciimg' in fname:
        return db.ScienceImage.from_file(fname)
    elif 'mskimg' in fname:
        return db.MaskImage.from_file(fname)


if __name__ == '__main__':

    db.init_db()
    my_files = get_my_share_of_work('tars')
    tmpdir = Path('/global/cscratch1/sd/dgold/inventory')

    for file in my_files:
        mydir = tmpdir / uuid.uuid4().hex
        os.chdir(mydir)
        cmd = f'/usr/common/mss/bin/hsi get {file}'
        try:
            subprocess.check_call(cmd.split())
        except subprocess.CalledProcessError:
            continue
        tar = tarfile.open(os.path.basename(file))
        tar.extractall()
        for member in tar:
            basename = os.path.basename(member.name)
            obj = db.DBSession().query(db.PipelineProduct).filter(
                db.PipelineProduct.basename == basename).first()
            if obj is None:
                obj = object_from_filename(basename)

            db.DBSession().add(obj)
        tar.close()
        os.remove(os.path.basename(file))
        db.DBSession().commit()


