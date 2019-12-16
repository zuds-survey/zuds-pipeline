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

        fullnames = []
        basenames = []
        for member in tar:
            fullnames.append(member.name)  # keep the full path here
            member.name = os.path.basename(member.name)
            basenames.append(member.name)
        tar.extractall()

        # need to do all queries first to avoid tripping sa autoflush (i.e.,
        # writing things to db before commit is issued)

        objects = db.DBSession().query(db.PipelineProduct.basename,
                                       db.PipelineProduct.id).filter(
            db.PipelineProduct.basename.in_(basenames)
        ).all()
        objects = dict(objects)

        tapearchive = db.TapeArchive(id=file, size=size)
        copies = []
        for (i, member), fullname in zip(enumerate(tar), fullnames):
            if 'HTAR_CF_CHK' in member.name:
                continue
            try:
                objid = objects[member.name]
                copy = db.TapeCopy(product_id=objid, archive=tapearchive,
                                   member_name=fullname)

            except KeyError:
                obj = object_from_filename(member.name)
                copy = db.TapeCopy(product=obj, archive=tapearchive,
                                   member_name=fullname)

            os.remove(member.name)
            copies.append(copy)
        tar.close()

        db.DBSession().add_all(copies)
        db.DBSession().commit()

        os.chdir(olddir)
        shutil.rmtree(mydir)




