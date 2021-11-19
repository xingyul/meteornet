


import os
import numpy as np
import tensorflow as tf
import six

def get_savename_from_varname(
        varname, varname_prefix=None,
        savename_prefix=None):
    """
    Args:
        varname(str): a variable name in the graph
        varname_prefix(str): an optional prefix that may need to be removed in varname
        savename_prefix(str): an optional prefix to append to all savename
    Returns:
        str: the name used to save the variable
    """
    name = varname
    if varname_prefix is not None \
            and name.startswith(varname_prefix):
        name = name[len(varname_prefix) + 1:]
    if savename_prefix is not None:
        name = savename_prefix + '/' + name
    return name

def get_checkpoint_path(model_path, logger):
    """
    Work around TF problems in checkpoint path handling.
    Args:
        model_path: a user-input path
    Returns:
        str: the argument that can be passed to NewCheckpointReader
    """
    if os.path.basename(model_path) == model_path:
        model_path = os.path.join('.', model_path)  # avoid #4921 and #6142
    if os.path.basename(model_path) == 'checkpoint':
        assert tf.gfile.Exists(model_path), model_path
        model_path = tf.train.latest_checkpoint(os.path.dirname(model_path))
        # to be consistent with either v1 or v2

    # fix paths if provided a wrong one
    new_path = model_path
    if '00000-of-00001' in model_path:
        new_path = model_path.split('.data')[0]
    elif model_path.endswith('.index'):
        new_path = model_path.split('.index')[0]
    if new_path != model_path:
        logger(
            "Checkpoint path {} is auto-corrected to {}.".format(model_path, new_path))
        model_path = new_path
    assert tf.gfile.Exists(model_path) or tf.gfile.Exists(model_path + '.index'), model_path
    return model_path

def is_training_name(name):
    """
    **Guess** if this variable is only used in training.
    Only used internally to avoid too many logging. Do not use it.
    """
    # TODO: maybe simply check against TRAINABLE_VARIABLES and MODEL_VARIABLES?
    # TODO or use get_slot_names()
    name = get_op_tensor_name(name)[0]
    if name.endswith('/Adam') or name.endswith('/Adam_1'):
        return True
    if name.endswith('/Momentum'):
        return True
    if name.endswith('/Adadelta') or name.endswith('/Adadelta_1'):
        return True
    if name.endswith('/RMSProp') or name.endswith('/RMSProp_1'):
        return True
    if name.endswith('/Adagrad'):
        return True
    if name.startswith('EMA/'):  # all the moving average summaries
        return True
    if name.startswith('AccumGrad') or name.endswith('/AccumGrad'):
        return True
    return False


def get_op_tensor_name(name):
    """
    Will automatically determine if ``name`` is a tensor name (ends with ':x')
    or a op name.
    If it is an op name, the corresponding tensor name is assumed to be ``op_name + ':0'``.

    Args:
        name(str): name of an op or a tensor
    Returns:
        tuple: (op_name, tensor_name)
    """
    if len(name) >= 3 and name[-2] == ':':
        return name[:-2], name
    else:
        return name, name + ':0'

class CheckpointReaderAdapter(object):
    """
    An adapter to work around old checkpoint format, where the keys are op
    names instead of tensor names (with :0).
    """
    def __init__(self, reader):
        self._reader = reader
        m = self._reader.get_variable_to_shape_map()
        self._map = {k if k.endswith(':0') else k + ':0': v
                     for k, v in six.iteritems(m)}

    def get_variable_to_shape_map(self):
        return self._map

    def get_tensor(self, name):
        if self._reader.has_tensor(name):
            return self._reader.get_tensor(name)
        if name in self._map:
            assert name.endswith(':0'), name
            name = name[:-2]
        return self._reader.get_tensor(name)

    def has_tensor(self, name):
        return name in self._map

    # some checkpoint might not have ':0'
    def get_real_name(self, name):
        if self._reader.has_tensor(name):
            return name
        assert self.has_tensor(name)
        return name[:-2]

class MismatchLogger(object):
    def __init__(self, exists, nonexists, logger):
        self._exists = exists
        self._nonexists = nonexists
        self._names = []
        self.logger = logger

    def add(self, name):
        self._names.append(name)

    def log(self):
        if len(self._names):
            self.logger("The following variables are in the {}, but not found in the {}: {}".format(
                self._exists, self._nonexists, ', '.join(self._names)))

class SaverRestore(object):
    """
    Restore a tensorflow checkpoint saved by :class:`tf.train.Saver` or :class:`ModelSaver`.
    """
    def __init__(self, model_path, logger, prefix=None, ignore=[]):
        """
        Args:
            model_path (str): a model name (model-xxxx) or a ``checkpoint`` file.
            prefix (str): during restore, add a ``prefix/`` for every variable in this checkpoint.
            ignore (list[str]): list of tensor names that should be ignored during loading, e.g. learning-rate
        """
        if model_path.endswith('.npy') or model_path.endswith('.npz'):
            logger("SaverRestore expect a TF checkpoint, but got a model path '{}'.".format(model_path) +
                        " To load from a dict, use 'DictRestore'.")
        model_path = get_checkpoint_path(model_path, logger)
        self.path = model_path  # attribute used by AutoResumeTrainConfig!
        self.prefix = prefix
        self.ignore = [i if i.endswith(':0') else i + ':0' for i in ignore]
        self.logger = logger

    def _setup_graph(self):
        dic = self._get_restore_dict()
        self.saver = tf.train.Saver(var_list=dic, name=str(id(dic)))

    def run_init(self, sess):
        self.logger("Restoring checkpoint from {} ...".format(self.path))
        self._setup_graph()
        self.saver.restore(sess, self.path)

    @staticmethod
    def _read_checkpoint_vars(model_path):
        """ return a set of strings """
        reader = tf.train.NewCheckpointReader(model_path)
        reader = CheckpointReaderAdapter(reader)    # use an adapter to standardize the name
        ckpt_vars = reader.get_variable_to_shape_map().keys()
        return reader, set(ckpt_vars)

    def _match_vars(self, func):
        reader, chkpt_vars = SaverRestore._read_checkpoint_vars(self.path)
        graph_vars = tf.global_variables()
        chkpt_vars_used = set()

        mismatch = MismatchLogger('graph', 'checkpoint', self.logger)
        for v in graph_vars:
            name = get_savename_from_varname(v.name, varname_prefix=self.prefix)
            if name in self.ignore and reader.has_tensor(name):
                self.logger("Variable {} in the graph will not be loaded from the checkpoint!".format(name))
            else:
                if reader.has_tensor(name):
                    func(reader, name, v)
                    chkpt_vars_used.add(name)
                else:
                    vname = v.op.name
                    if not is_training_name(vname):
                        mismatch.add(vname)
        mismatch.log()
        mismatch = MismatchLogger('checkpoint', 'graph', self.logger)
        if len(chkpt_vars_used) < len(chkpt_vars):
            unused = chkpt_vars - chkpt_vars_used
            for name in sorted(unused):
                if not is_training_name(name):
                    mismatch.add(name)
        mismatch.log()

    def _get_restore_dict(self):
        var_dict = {}

        def f(reader, name, v):
            name = reader.get_real_name(name)
            assert name not in var_dict, "Restore conflict: {} and {}".format(v.name, var_dict[name].name)
            var_dict[name] = v
        self._match_vars(f)
        return var_dict
