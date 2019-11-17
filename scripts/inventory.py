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
        size = os.path.getsize(tar.name)

        for member in tar:
            member.name = os.path.basename(member.name)
        tar.extractall()

        # need to do all queries first to avoid tripping sa autoflush (i.e.,
        # writing things to db before commit is issued)
        objects = []
        for member in tar:
            basename = member.name
            obj = db.DBSession().query(db.PipelineProduct).filter(
                db.PipelineProduct.basename == basename).first()
            objects.append(obj)

        tapearchive = db.TapeArchive(id=file, size=size)
        copies = []
        for i, member in enumerate(tar):
            obj = objects[i]
            if obj is None:
                obj = object_from_filename(member.name)
            copy = db.TapeCopy(product=obj, archive=tapearchive)
            os.remove(member.name)
            copies.append(copy)
        tar.close()

        db.DBSession().add_all(copies)
        db.DBSession().commit()

        os.chdir(olddir)
        shutil.rmtree(mydir)




