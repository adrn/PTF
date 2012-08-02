#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

/* Define docstrings */
static char module_docstring[] = "Fast integration";
static char constant_docstring[] = "Constant function";
static char linear_docstring[] = "Linear function";
static char gaussian_docstring[] = "Gaussian function";

/* Declare the C functions here. */
static PyObject *gaussian_error_func(PyObject *self, PyObject *args);
static PyObject *constant_error_func(PyObject *self, PyObject *args);
static PyObject *linear_error_func(PyObject *self, PyObject *args);

/* Define the methods that will be available on the module. */
static PyMethodDef module_methods[] = {
    {"gaussian_error_func", gaussian_error_func, METH_VARARGS, gaussian_docstring},
    {"constant_error_func", constant_error_func, METH_VARARGS, constant_docstring},
    {"linear_error_func", linear_error_func, METH_VARARGS, linear_docstring},
    {NULL, NULL, 0, NULL}
};

/* This is the function that is called on import. */

#if PY_MAJOR_VERSION >= 3
  #define MOD_ERROR_VAL NULL
  #define MOD_SUCCESS_VAL(val) val
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
          ob = PyModule_Create(&moduledef);
#else
  #define MOD_ERROR_VAL
  #define MOD_SUCCESS_VAL(val)
  #define MOD_INIT(name) void init##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          ob = Py_InitModule3(name, methods, doc);
#endif

MOD_INIT(error_functions)
{
    PyObject *m;
    MOD_DEF(m, "error_functions", module_docstring, module_methods);
    if (m == NULL)
        return MOD_ERROR_VAL;
    import_array();
    return MOD_SUCCESS_VAL(m);
}

/* Do the heavy lifting here */

static PyObject *gaussian_error_func(PyObject *self, PyObject *args)
{

    double p0, p1, p2, p3;
    PyObject *p, *x_obj, *y_obj, *e_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "(dddd)OOO", &p0, &p1, &p2, &p3, &x_obj, &y_obj, &e_obj))
        return NULL;
    
    /* Interpret the input objects as `numpy` arrays. */
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *e_array = PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an `Exception`. */
    if (x_array == NULL || y_array == NULL || e_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        Py_XDECREF(e_array);
        return NULL;
    }

    /* How many data points are there? */
    int n = (int)PyArray_DIM(x_array, 0);

    /* Check the dimensions. */
    if (n != (int)PyArray_DIM(y_array, 0) || n != (int)PyArray_DIM(e_array, 0)) {
        PyErr_SetString(PyExc_RuntimeError, "Dimension mismatch.");
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        Py_DECREF(e_array);
        return NULL;
    }

    /* Build the output array */
    npy_intp dims[1];
    dims[0] = n;
    PyObject *r_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (r_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't build output array");
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        Py_XDECREF(r_array);
        return NULL;
    }

    /* Get pointers to the data as C-types. */
    double *x = (double*)PyArray_DATA(x_array);
    double *y = (double*)PyArray_DATA(y_array);
    double *e = (double*)PyArray_DATA(e_array);
    double *r = (double*)PyArray_DATA(r_array);

    /* Calculate error function */

    int i;
    double d;

    for(i = 0; i < n; i++) {
        d = (x[i] - p1) / p2;
        r[i] = (y[i] - p0 * exp(- 0.5 * d * d) - p3) / e[i];
    }

    /* Clean up. */
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(e_array);

    return r_array;

}

static PyObject *constant_error_func(PyObject *self, PyObject *args)
{

    double p0;
    PyObject *p, *x_obj, *y_obj, *e_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "(d)OOO", &p0, &x_obj, &y_obj, &e_obj))
        return NULL;
    
    /* Interpret the input objects as `numpy` arrays. */
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *e_array = PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an `Exception`. */
    if (x_array == NULL || y_array == NULL || e_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        Py_XDECREF(e_array);
        return NULL;
    }

    /* How many data points are there? */
    int n = (int)PyArray_DIM(x_array, 0);

    /* Check the dimensions. */
    if (n != (int)PyArray_DIM(y_array, 0) || n != (int)PyArray_DIM(e_array, 0)) {
        PyErr_SetString(PyExc_RuntimeError, "Dimension mismatch.");
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        Py_DECREF(e_array);
        return NULL;
    }

    /* Build the output array */
    npy_intp dims[1];
    dims[0] = n;
    PyObject *r_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (r_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't build output array");
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        Py_XDECREF(r_array);
        return NULL;
    }

    /* Get pointers to the data as C-types. */
    double *x = (double*)PyArray_DATA(x_array);
    double *y = (double*)PyArray_DATA(y_array);
    double *e = (double*)PyArray_DATA(e_array);
    double *r = (double*)PyArray_DATA(r_array);

    /* Calculate error function */

    int i;

    for(i = 0; i < n; i++) {
        r[i] = (y[i] - p0) / e[i];
    }

    /* Clean up. */
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(e_array);

    return r_array;

}

static PyObject *linear_error_func(PyObject *self, PyObject *args)
{

    double p0, p1;
    PyObject *p, *x_obj, *y_obj, *e_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "(dd)OOO", &p0, &p1, &x_obj, &y_obj, &e_obj))
        return NULL;
    
    /* Interpret the input objects as `numpy` arrays. */
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *e_array = PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an `Exception`. */
    if (x_array == NULL || y_array == NULL || e_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        Py_XDECREF(e_array);
        return NULL;
    }

    /* How many data points are there? */
    int n = (int)PyArray_DIM(x_array, 0);

    /* Check the dimensions. */
    if (n != (int)PyArray_DIM(y_array, 0) || n != (int)PyArray_DIM(e_array, 0)) {
        PyErr_SetString(PyExc_RuntimeError, "Dimension mismatch.");
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        Py_DECREF(e_array);
        return NULL;
    }

    /* Build the output array */
    npy_intp dims[1];
    dims[0] = n;
    PyObject *r_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (r_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't build output array");
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        Py_XDECREF(r_array);
        return NULL;
    }

    /* Get pointers to the data as C-types. */
    double *x = (double*)PyArray_DATA(x_array);
    double *y = (double*)PyArray_DATA(y_array);
    double *e = (double*)PyArray_DATA(e_array);
    double *r = (double*)PyArray_DATA(r_array);

    /* Calculate error function */

    int i;

    for(i = 0; i < n; i++) {
        r[i] = (y[i] - p0*x[i] - p1) / e[i];
    }

    /* Clean up. */
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(e_array);

    return r_array;

}