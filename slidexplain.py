# -*- coding: utf-8 -*-
import ipywidgets as widgets #migrated
from IPython.core.display import display
import pandas as pd
from scipy.optimize import minimize, curve_fit
import numpy as np
from collections import defaultdict
import inspect
import numbers
import itertools as itt
import seaborn as sns
import math
import sys


def _generate_widget(boundaries, description, width=0.8, step=0.001):
    bound_min, bound_max = boundaries
    if type(bound_min) == float or type(bound_max) == float:
        new_widget = widgets.FloatSlider(
            description=description,
            min=bound_min,
            max=bound_max,
            step=step,
            layout=widgets.Layout(width=f'{width*100}%'))
    elif type(bound_min) == int or type(bound_max) == int:
        new_widget = widgets.IntSlider(
            description=description,
            min=bound_min,
            max=bound_max,
            step=step,
            layout=widgets.Layout(width=f'{width*100}%'))
    elif type(bound_min) == bool or type(bound_max) == bool:
        new_widget = widgets.Checkbox(
            description=description,
            min=bound_min,
            max=bound_max,
            step=step,
            layout=widgets.Layout(width=f'{width*100}%'))
    return new_widget


class FunctionExplainer(object):
    '''
    Object used to interact with functions on a Jupyter Notebook and
    understand its properties.

    Parameters
    ----------
    title_text : Title showed when you display the explainer.
    width : Width of the explainer.
    step : Minimum value used to update the value of the sliders.
    
    Examples
    --------
    Radius of a circle.

    explainer = sle.FunctionExplainer('Radius of a circle')
    explainer.display(
        lambda x, y: x**2 + y**2,
        boundaries={'x': (-1., 1.), 'y': (-1., 1.)},
        target_boundaries=(0., 2.))
    '''

    def __init__(self, title_text='', width=0.8, step=0.001):
        self.title_text = title_text
        self.width = width
        self.step = step
    
    def display(
        self, function, target_boundaries,
        boundaries, kw_names=None):
        '''
        Display the explainer for a given function.

        Parameters
        ----------
        function : Function to interact with.
        boundaries : Dictionary containing a tuple of 2 elements for
            parameters of the function, with the boundaries of the slider
            generated for each parameter.
        target_boundaries : A tuple of 2 values with the boundaries for
            the result returned by the function.
        kw_names : If the function uses keyworded variable-length argument list,
            you must pass its names in kw_names.
        '''
        fspec = inspect.getfullargspec(function)
        arguments = fspec.args
        kw_name = fspec.varkw
        if len(arguments) == 0 and kw_name is not None:
            arguments = kw_names

        # Generate widgets.
        caption = widgets.Label(
            value=self.title_text,
            layout=widgets.Layout(
                display='flex',
                flex_flow='column',
                align_items='center'))
        widget_dic = {}
        for a in arguments:
            bound_min, bound_max = boundaries.get(
                a, (sys.float_info.min, sys.float_info.max))
            new_widget = _generate_widget(
                boundaries=(bound_min, bound_max),
                description=a,
                width=self.width,
                step=self.step,
                )
            widget_dic[a] = new_widget
        bound_min, bound_max = target_boundaries
        result_widget = _generate_widget(
            boundaries=(bound_min, bound_max),
            description='<b>Result</b>',
            width=self.width,
            step=self.step,
            )

        # Connect widgets to target result.
        for wid in widget_dic.values():
            #target_values = self.boundaries.copy()
            widgets.dlink(
                (wid, 'value'), (result_widget, 'value'),
                transform=lambda v: function(**{x: widget_dic[x].value for x in arguments}),
                )
        wid_list = [w for w in widget_dic.values()]
        wid_list.append(result_widget)
        display(caption, *wid_list)


class DataFrameExplainer(FunctionExplainer):
    '''
    Object used to interact with pandas DataFrame objects on
    a Jupyter Notebook and understand its properties.

    Parameters
    ----------
    title_text : Title showed when you display the explainer.
    width : Width of the explainer.
    step : Minimum value used to update the value of the sliders.
    
    Examples
    --------
    df = pd.DataFrame({
        'a': [1., 5., 7., 2., -1., 9.],
        'b': [1., 2., 6., -2., -1., 1.]})
    y = 0.2*df['a']**2 + 0.8*df['a'] - 1.2*df['b']

    explainer = sle.DataFrameExplainer('Fitting a dataframe.')
    explainer.display(
        df, y, max_degree=2,
        boundaries={'a': (-1., 1.), 'b': (-1., 1.)},
        target_boundaries=(-10., 10.))
    '''
    def __init__(self, title_text='', width=0.8, step=0.001):
        super().__init__(title_text=title_text, width=width, step=step)

    def fit_function_to_data(self, df_features, y, max_degree=1):
        X = np.hstack([
            df_features**(i+1)
                for i in range(max_degree)])
        def func(X, *params):
            return np.hstack(params).dot(X)
        popt, _ = curve_fit(func, X.T, y, p0=np.ones(X.shape[1]))
        return popt
    
    def generate_function_from_data(
        self, df_features, y, max_degree=1):
        coefs = self.fit_function_to_data(
            df_features, y, max_degree=max_degree)
        
        # Function fitted to the data.
        def fun(**kw):
            values = pd.Series(kw)
            values = np.hstack([
                values**(i+1)
                    for i in range(max_degree)])
            return coefs.dot(values)
        return fun
    
    def display(
        self, df_features, y, max_degree=1,
        target_boundaries=None, boundaries=None):
        '''
        Display the explainer for a given dataframe.

        Parameters
        ----------
        function : DataFrame to interact with.
        boundaries : Dictionary containing a tuple of 2 elements for
            parameters of the function, with the boundaries of the slider
            generated for each parameter. If None, it will use the minimum
            and maximum values found in the dataframe.
        target_boundaries : A tuple of 2 values with the boundaries for
            the result returned by the function. If None, it will use the
            minimum and maximum value found in the dataframe.
        max_degree : Degree of the polynomial fitted to the data to explain
            its relationships.
        '''
        if target_boundaries is None:
            target_boundaries = (y.min(), y.max())
        if boundaries is None:
            min_list, max_list = df.min(), df.max()
            boundaries = {k: (min_list[k], max_list[k]) for k in df.columns}
        fitted_function = self.generate_function_from_data(
            df_features, y, max_degree=max_degree)
        super().display(
            fitted_function,
            target_boundaries=target_boundaries,
            boundaries=boundaries,
            kw_names=list(df_features.columns))