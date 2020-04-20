import sqlalchemy as sa
from pathlib import Path


__all__ = ['UnmappedFileError', 'File']

class UnmappedFileError(FileNotFoundError):
    """Error raised when a user attempts to call a method of a `File` that
    requires the file to be mapped to a file on disk, but the file is not
    mapped. """
    pass


class File(object):
    """A python object mappable to a file on spinning disk, with metadata
    mappable to rows in a database.`File`s should be thought of as python
    objects that live in memory that can represent the data and metadata of
    files on disk. If mapped to database records, they can serve as an
    intermediary between database records and files that live on disk.

    `File`s can read and write data and metadata to and from disk and to and
    from the database. However, there are some general rules about this that
    should be heeded to ensure good performance.

    In general, files that live on disk and are represented by `File`s
    contain data that users and subclasses may want to use. It is imperative
    that file metadata, and only file metadata, should be mapped to the
    database by instances of File. Data larger than a few kb from disk-mapped
    files should never be stored in the database. Instead it should reside on
    disk. Putting the file data directly into the database would slow it down
    and make queries take too long.

    Files represented by this class can reside on spinning disk only. This
    class does not make any attempt to represent files that reside on tape.

    The user is responsible for manually associating `File`s  with files on
    disk. This frees `File`s  from being tied to specific blocks of disk on
    specific machines.  The class makes no attempt to enforce that the user
    maps python objects tied to particular database records to the "right"
    files on disk. This is the user's responsibility!

    `property`s of Files should be assumed to represent what is in memory
    only, not necessarily what is on disk. Disk-memory synchronization is up
    to the user and can be achived using the save() function.
    """

    basename = sa.Column(sa.Text, unique=True, index=True)
    __diskmapped_cached_properties__ = ['_path']

    @property
    def local_path(self):
        try:
            return self._path
        except AttributeError:
            errormsg = f'File "{self.basename}" is not mapped to the local ' \
                       f'file system. ' \
                       f'Identify the file corresponding to this object on ' \
                       f'the local file system, ' \
                       f'then call the `map_to_local_file` to identify the ' \
                       f'path. '
            raise UnmappedFileError(errormsg)

    @property
    def ismapped(self):
        return hasattr(self, '_path')

    def map_to_local_file(self, path, quiet=False):
        if not quiet:
            if  hasattr(self, 'basename'):
                print(f'Mapping {self.basename} to {path}')
            else:
                print(f'Mapping {self} to {path}')
        self._path = str(Path(path).absolute())

    def unmap(self):
        if not self.ismapped:
            raise UnmappedFileError(f"Cannot unmap file '{self.basename}', "
                                    f"file is not mapped")
        self.clear()

    def clear(self):
        for attr in self.__diskmapped_cached_properties__:
            if hasattr(self, attr):
                delattr(self, attr)

    def save(self):
        """Update the data and metadata of a mapped file on disk to reflect
        their values in this object."""
        raise NotImplemented

    def load(self):
        """Load the data and metadata of a mapped file on disk into memory
        and set the values of database mapped columns, which can later be
        flushed into the DB."""
        raise NotImplemented
