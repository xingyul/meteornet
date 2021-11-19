


import os
import numpy as np
import tensorflow as tf
import six



class SessionUpdate(object):
    """ Update the variables in a session """

    def __init__(self, sess, vars_to_update, logger):
        """
        Args:
            sess (tf.Session): a session object
            vars_to_update: a collection of variables to update
        """
        self.sess = sess
        self.name_map = {v.name: v for v in vars_to_update}
        self.logger = logger

    @staticmethod
    def load_value_to_var(var, val, logger, strict=False):
        """
        Call `var.load(val)` with the default session.

        Args:
            var (tf.Variable):
            strict (bool): Behave less strict if set to False.
        """
        if strict:
            var.load(val)
            return
        name = var.op.name

        # check incompatible shape
        varshape = tuple(var.get_shape().as_list())
        if varshape != val.shape:
            # TODO only allow reshape when shape different by empty axis
            assert np.prod(varshape) == np.prod(val.shape), \
                "{}: {}!={}".format(name, varshape, val.shape)
            logger("Variable {} is reshaped {}->{} during assigning".format(
                name, val.shape, varshape))
            val = val.reshape(varshape)

        # fix some common type incompatibility problems, but not all
        def upcast(vartype, valtype):
            # allow up-casting
            if vartype == tf.float64 and valtype == np.float32:
                return np.float64
            if vartype in [tf.int64, tf.int32] and valtype in [np.int32, np.int16, np.int8]:
                return np.int64 if vartype == tf.int64 else np.int32
            return None

        if hasattr(val, 'dtype'):
            vartype = var.value().dtype
            if vartype != val.dtype:
                msg = "Variable {} has dtype {} but was given a value of dtype {}.".format(name, vartype, val.dtype)
                newtype = upcast(var.dtype, val.dtype)
                if newtype is not None:
                    val = newtype(val)
                    logger(msg + " Load it after casting!")
                else:
                    assert vartype == val.dtype, msg
        try:
            var.load(val)
        except tf.errors.InvalidArgumentError:
            logger("Cannot load this value to the variable {}".format(name))

    def update(self, prms):
        """
        Args:
            prms(dict): dict of {variable name: value}
                Any name in prms must be in the graph and in vars_to_update.
        """
        with self.sess.as_default():
            for name, value in six.iteritems(prms):
                assert name in self.name_map
                v = self.name_map[name]
                SessionUpdate.load_value_to_var(v, value, self.logger)


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

class DictRestore(object):
    """
    Restore variables from a dictionary.
    """

    def __init__(self, variable_dict, logger):
        """
        Args:
            variable_dict (dict): a dict of {name: value}
        """
        assert isinstance(variable_dict, dict), type(variable_dict)
        # use varname (with :0) for consistency
        self._prms = {get_op_tensor_name(n)[1]: v for n, v in six.iteritems(variable_dict)}
        self.logger = logger

    def run_init(self, sess):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        variable_names = set([k.name for k in variables])
        param_names = set(six.iterkeys(self._prms))

        intersect = variable_names & param_names

        self.logger("Variables to restore from dict: {}".format(', '.join(map(str, intersect))))

        mismatch = MismatchLogger('graph', 'dict', self.logger)
        for k in sorted(variable_names - param_names):
            if not is_training_name(k):
                mismatch.add(k)
        mismatch.log()
        mismatch = MismatchLogger('dict', 'graph', self.logger)
        for k in sorted(param_names - variable_names):
            mismatch.add(k)
        mismatch.log()

        upd = SessionUpdate(sess, [v for v in variables if v.name in intersect], self.logger)
        upd.update({name: value for name, value in six.iteritems(self._prms) if name in intersect})

