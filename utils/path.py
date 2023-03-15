"""
Directory configuration.

Author: Gijs G. Hendrickx
"""
import logging
import os

LOG = logging.getLogger(__name__)


class DirConfig:
    """Configuring directories and files in a robust and flexible manner, allowing to store a common working directory
    to/from multiple files are exported/imported.
    """
    __base_dirs = ('', 'C:', 'D:', 'P:', 'U:')

    def __init__(self, *home_dir, create_dir=True):
        """
        :param home_dir: home directory, defaults to None
        :param create_dir: automatically create home directory if non-existent, defaults to True

        :type home_dir: DirConfig, str, iterable[str]
        :type create_dir: bool, optional
        """
        self._home = self._unpack(home_dir)
        if create_dir:
            self.create_dir(self)

    def __repr__(self):
        """Representation of DirConfig."""
        return self._list2str(self._home_dir)

    @property
    def _sep(self):
        """Folder separator."""
        return os.sep

    @property
    def _current_dir(self):
        """Current directory.

        :return: directory
        :rtype: list[str]
        """
        return self._as_list(os.getcwd())

    @property
    def _home_dir(self):
        """Absolute home directory, set to current directory if no absolute directory is provided.

        :return: directory
        :rtype: list[str]
        """
        if not self._home:
            return self._current_dir

        list_dir = self._as_list(
            os.path.dirname(self._home) if os.path.splitext(self._home)[1] else self._home
        )
        return self._dir2abs(list_dir)

    def _unpack(self, directory):
        """Unpack defined directory, which may be a mix of str, tuple, list, and/or DirConfig.

        :param directory: defined directory
        :type directory: iterable

        :return: directory
        :rtype: str
        """
        out = []
        for item in directory:
            # directory is DirConfig
            if isinstance(item, type(self)):
                out.append(str(item))

            # directory is str
            elif isinstance(item, str):
                out.append(str(item))

            # directory is iterable[str]
            elif isinstance(item, (tuple, list)):
                out.append(self._unpack(item))

        # return directory
        return self._list2str(out)

    @staticmethod
    def _str2list(str_dir):
        """Translate string- to list-directory.

        :param str_dir: string-based directory
        :type str_dir: str

        :return: directory
        :rtype: list[str]
        """
        return str_dir.replace('/', '\\').split('\\')

    def _as_list(self, folder):
        """Ensure directory to be a list.

        :param folder: directory to be checked
        :type folder: str, list, tuple

        :return: directory
        :rtype: list[str]
        """
        if isinstance(folder, (str, DirConfig)):
            return self._str2list(str(folder))

        elif isinstance(folder, (list, tuple)):
            list_dir = []
            [list_dir.extend(self._str2list(i)) for i in folder]
            return list_dir

        else:
            msg = f'Directory must be str, list, or tuple; {type(folder)} is given.'
            raise TypeError(msg)

    def _list2str(self, list_dir):
        """Translate list- to string-directory.

        :param list_dir: list-based directory
        :type list_dir: list

        :return: directory
        :rtype: str
        """
        return self._sep.join(list_dir)

    def _dir2abs(self, folder):
        """Translate directory to absolute directory.

        :param folder: directory to be converted
        :type folder: list

        :return: absolute directory
        :rtype: list[str]
        """
        if self._is_abs_dir(folder):
            return folder
        return [*self._current_dir, *folder]

    def _is_abs_dir(self, folder):
        """Verify if directory is an absolute directory.

        :param folder: directory to be verified
        :type folder: list[str]

        :return: directory is an absolute directory, or not
        :rtype: bool
        """
        return folder[0] in self.__base_dirs

    def config_dir(self, *file_dir, relative_dir=False):
        """Configure directory.

        :param file_dir: file directory
        :param relative_dir: directory as relative directory, defaults to True

        :type file_dir: DirConfig, str, iterable[str]
        :type relative_dir: bool, optional

        :return: (absolute) directory
        :rtype: str
        """
        # unpack folder(s) (and file)
        list_dir = self._as_list(self._unpack(file_dir))

        # return listed folder(s) (and file)
        if self._is_abs_dir(list_dir) or relative_dir:
            return self._list2str(list_dir)

        # return home directory with listed folder(s) (and file)
        return self._list2str([*self._home_dir, *list_dir])

    def create_dir(self, *file_dir):
        """Configure and create directory, if non-existing.

        :param file_dir: (file) directory to be created
        :type file_dir: DirConfig, str, iterable[str]
        """
        # define directory
        file_dir = self.config_dir(file_dir)

        # create directory
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
            LOG.info(f'Directory created\t:\t{file_dir}')

        # return directory
        return file_dir

    def delete_file(self, *file_dir):
        """Delete file, if existing.

        :param file_dir: file directory
        :type file_dir: DirConfig, str, iterable[str]
        """
        # define directory
        file_dir = self.config_dir(file_dir)

        # delete file
        if self.existence_file(file_dir):
            os.remove(file_dir)
            LOG.info(f'File deleted\t:\t{file_dir}')

    def existence_file(self, *file_dir):
        """Verify if file exists.

        :param file_dir: file directory
        :type file_dir: DirConfig, str, iterable[str]

        :return: existence of file (directory)
        :rtype: bool
        """
        # define file directory
        file_dir = self.config_dir(file_dir)

        # check existence
        if os.path.exists(file_dir):
            # > file exists
            LOG.info(f'File exists\t:\t{file_dir}')
            return True

        # > file does not exist
        LOG.warning(f'File does not exist\t:\t{file_dir}')
        return False
