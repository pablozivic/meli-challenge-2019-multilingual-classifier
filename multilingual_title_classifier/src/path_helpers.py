import os
import multilingual_title_classifier

PACKAGE_PATH = os.path.abspath(os.path.dirname(multilingual_title_classifier.__file__))
RESOURCES_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, '..', 'resources'))


def get_resources_path(filename) -> str:
    path = os.path.join(RESOURCES_PATH, filename)
    dirs = '/'.join(path.split('/')[:-1])

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    return path


def get_path(*args, dirs=None, **kwargs) -> str:
    path = []
    for arg in args:
        path.append(str(arg))
    for k, v in kwargs.items():
        if isinstance(v, bool):
            if v:
                path.append(k)
        else:
            path.append('{}_{}'.format(k, v))

    dirs_str = ''
    if dirs:
        dirs_str = '/'.join(dirs) + '/'

    return get_resources_path(dirs_str + '_'.join(path))
