import os
import db
import subprocess
import tarfile
import uuid
import shutil
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
    olddir = os.getcwd()

    for file in my_files:
        mydir = tmpdir / uuid.uuid4().hex
        mydir.mkdir()
        os.chdir(mydir)
        cmd = f'/usr/common/mss/bin/hsi get {file}'
        try:
            subprocess.check_call(cmd.split())
        except subprocess.CalledProcessError:
            continue
        tar = tarfile.open(os.path.basename(file))
        tar.extractall()

        tapearchive = db.TapeArchive(id=file)

        for member in tar:
            basename = os.path.basename(member.name)
            obj = db.DBSession().query(db.PipelineProduct).filter(
                db.PipelineProduct.basename == basename).first()
            if obj is None:
                obj = object_from_filename(basename)

            copy = db.TapeCopy(product=obj, archive=tapearchive)
            db.DBSession().add(copy)

        tar.close()
        db.DBSession().commit()
        os.chdir(olddir)
        shutil.rmtree(mydir)




