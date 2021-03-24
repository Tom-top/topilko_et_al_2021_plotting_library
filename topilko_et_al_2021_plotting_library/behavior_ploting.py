#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:08:24 2021

@author: tomtop
"""

import os

import math
import numpy as np
import pandas as pd
from natsort import natsorted
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from scipy import stats

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


class BehaviorException(Exception):
    pass


# class BarPlot(Plotting):
#     def __init__(self):
#         super(BarPlot, self).__init__()
#     def


class Plotting(object):
    def __init__(self):
        ######################################################################
        # Parameters for plot sizing
        ######################################################################

        self._cm_to_inch = 25.4
        self._inch_to_cm = 1 / self._cm_to_inch

        ######################################################################
        # Parameters for plot fonts
        ######################################################################

        self._fontsize_label = 3.
        self._fontsize_title = 3.

        ######################################################################
        # Parameters for plot spacing
        ######################################################################

        self._y_top_offset = 0.4
        self._y_bottom_offset = 0.025

        ######################################################################
        # Parameters for figure canvas
        ######################################################################

        self._linewidth_canvas = 0.25
        self._length_ticks = 1.
        self._width_ticks = 0.25
        self._pad_ticks = 0.5
        self._pad_labels = 1.

        ######################################################################
        # Parameters for plot graphics
        ######################################################################

        self._number_x_ticks = 5
        self._number_y_ticks = 5
        self._size_markers = 2.
        self._linewidth_markers = 0.25
        self._linewidth_plot = 0.25
        self._rotation_label = 45
        self._pval_shift = 0.05
        self._stat_method = "mannwhitneyu"
        self._verbose = True
        self._dist_sbars = 0.05

        self._barplot_kwargs = {"error_kw": {"elinewidth": self._linewidth_plot,
                                             "capsize": 1.,
                                             "capthick": self._linewidth_plot},
                                "width": 0.8,
                                "linewidth": self._linewidth_plot,
                                "edgecolor": "black",
                                "alpha": 1.,
                                "zorder": 0, }

        self._swarmplot_kwargs = {"linewidth": self._linewidth_markers,
                                  "s": self._size_markers,
                                  "edgecolor": "black",
                                  "alpha": 0.8,
                                  "zorder": 1, }

        self._cumulativeplot_meanline_kwargs = {"lw": 0.8,
                                                "alpha": 1, }

        self._cumulativeplot_stdline_kwargs = {"lw": 0.5,
                                               "alpha": 0.2, }

        ######################################################################
        # Parameters for plot colors
        ######################################################################

        self._behavior_color_map = {"nest": "#f97c7c",
                                    "sleep": "#84d9ff",
                                    "snifffood": "#77dd77",
                                    "drink": "#b19cd9",
                                    "dig": "#ffb347",
                                    "groom": "#f49ac2"}

    @staticmethod
    def stars_for_pvalue(pval):
        if pval > 0.05:
            return "n.s"
        elif 0.05 >= pval > 0.01:
            return "*"
        elif 0.01 >= pval > 0.001:
            return "**"
        elif 0.001 >= pval > 0.0001:
            return "***"
        elif 0.0001 >= pval:
            return "****"

    @staticmethod
    def get_statistical_method(method):
        try:
            fc = getattr(stats, method)
            return fc
        except:
            raise BehaviorException("method: {} is not available in module scipy.stats".format(method))

    def generate_pvals(self, data, combinations, method="mannwhitneyu"):
        fc = self.get_statistical_method(method)
        pvals = []
        for comb in combinations:
            pvals.append(fc(data.iloc[:, comb[0]][~data.iloc[:, comb[0]].isnull()],
                            data.iloc[:, comb[1]][~data.iloc[:, comb[1]].isnull()])[-1])
        return pvals

    @staticmethod
    def filter_pairs(arr):
        arr = np.array(arr)
        idx = np.lexsort((arr[:, 1], arr[:, 0]))
        return arr[idx]

    @staticmethod
    def check_identical_pair(source, target):
        cond = np.array(source) == np.array(target)
        if cond.all():
            return True
        else:
            return False

    @staticmethod
    def check_identical_values_in_pair(source):
        if source[0] == source[1]:
            return True
        else:
            return False

    def check_correct_pair(self, source, target):
        is_identical = self.check_identical_values_in_pair(source)
        if is_identical:
            raise RuntimeError("source pair is identical: {}".format(source))
        are_identical = self.check_identical_pair(source, target)
        if are_identical:
            raise RuntimeError("source and target pairs are identical: {}-{}".format(source, target))

    @staticmethod
    def check_pair_overlaps(pair0, pair1):
        intersection = range(max(pair0[0], pair1[0]),
                             min(pair0[-1], pair1[-1]))
        if len(intersection) != 0:
            return True
        else:
            return False

    def generate_height_from_combinations(self, combinations):
        sink = {}
        for pair in combinations:
            print("\n new pair")
            pair = list(pair)
            if not sink:
                sink[1] = []
                sink[1].append(pair)
            else:
                for length, value in sink.items():
                    print("checking new value")
                    general_overlap = 0
                    overlap = 0
                    for _pair in value:
                        self.check_correct_pair(pair, _pair)
                        pair_overlaps = self.check_pair_overlaps(pair, _pair)
                        overlap += pair_overlaps
                        general_overlap += pair_overlaps
                        print(pair, _pair, general_overlap)
                    if overlap == 0:
                        sink[length].append(pair)
                        break
                if general_overlap > 0:
                    print("new value! {}".format(length + 1))
                    sink[length + 1] = []
                    sink[length + 1].append(pair)
        sink = self.convert_height_dict_to_list(combinations, sink)
        return sink

    def convert_height_dict_to_list(self, combinations, height_dict):
        sink = []
        for c in combinations:
            for k, v in height_dict.items():
                if list(c) in v:
                    sink.append(k)
        return sink

    def display_significance(self, ax, pvals, combinations, y_max, y_min):
        y_range = abs(y_max - y_min)
        combinations = self.filter_pairs(combinations)
        heights = self.generate_height_from_combinations(combinations)
        for pval, x_pos, h in zip(pvals, combinations, heights):
            y_pos_low = y_max + (y_range * 0.1) * h
            y_pos_high = y_max + (y_range * 0.1) * h + (y_range * 0.02)
            y_pos_label = y_max + (y_range * 0.1) * h + (y_range * 0.025)
            ax.plot([x_pos[0] + self._dist_sbars, x_pos[1] - self._dist_sbars],
                    [y_pos_high, y_pos_high],
                    color="black",
                    lw=self._linewidth_plot)
            ax.plot([x_pos[0] + self._dist_sbars, x_pos[0] + self._dist_sbars],
                    [y_pos_low, y_pos_high],
                    color="black",
                    lw=self._linewidth_plot)
            ax.plot([x_pos[1] - self._dist_sbars, x_pos[1] - self._dist_sbars],
                    [y_pos_low, y_pos_high],
                    color="black",
                    lw=self._linewidth_plot)
            ax.text((x_pos[0] + x_pos[1]) / 2,
                    y_pos_label,
                    self.stars_for_pvalue(pval),
                    fontsize=self._fontsize_title,
                    ha="center")

    @staticmethod
    def is_nested_dict(source):
        return any(isinstance(i, dict) for i in source.values())

    @staticmethod
    def is_dict(source):
        return isinstance(source, dict)

    def format_figure_canvas(self, ax):
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(self._linewidth_canvas)
        ax.tick_params(length=self._length_ticks,
                       width=self._width_ticks,
                       axis='both',
                       which='major',
                       pad=self._pad_ticks)

    def format_box_plot_graphics(self, ax, colors):
        for i, box in enumerate(ax.artists):
            box.set_edgecolor('black')
            box.set_facecolor(colors[i])
            for j in range(5 * i, 5 * (i + 1)):
                ax.lines[j].set_color('black')
        self.format_figure_canvas(ax)

    def is_correct_data_type(self, source):
        is_dict = self.is_dict(source)
        if not is_dict:
            raise BehaviorException("data is of invalid type. dtype requiered: dict/nested dict")
        is_nested = self.is_nested_dict(source)
        if not is_nested:
            raise BehaviorException("data is of invalid type. dtype requiered: dict/nested dict")

    @staticmethod
    def get_tick_size_time(data, nticks=4):
        sect_size = data / nticks
        if sect_size >= 3600:
            label_divider = 3600
            remainder = sect_size % label_divider
            tick_size = sect_size - remainder
            tick_size_label = tick_size / label_divider
            unit = "hr"
        elif 3600 > sect_size >= 60:
            label_divider = 60
            remainder = sect_size % label_divider
            tick_size = sect_size - remainder
            tick_size_label = tick_size / label_divider
            unit = "min"
        else:
            label_divider = 1
            tick_size = sect_size
            tick_size_label = tick_size / label_divider
            unit = "s"
        return tick_size, tick_size_label, label_divider, unit

    def get_max_values_with_offset(self, ax_max, ax_min):
        ax_range = ax_max - ax_min
        ax_max_offset = ax_max + ax_range * self._y_top_offset
        ax_min_offset = ax_min - ax_range * self._y_bottom_offset
        return ax_max_offset, ax_min_offset

    def get_max_values(self, data):
        ax_max = np.max(data.max())
        ax_min = np.min(data.min())
        return ax_max, ax_min

    @staticmethod
    def get_max_values_without_offset(data):
        ax_max = np.max(data.max())
        ax_min = np.min(data.min())
        return ax_max, ax_min

    def generate_ticks_time(self, ax_max_offset, ax_min_offset):
        ax_range_offset = ax_max_offset - ax_min_offset
        ax_tick_size, ax_tick_size_label, label_divider, unit = self.get_tick_size_time(ax_range_offset,
                                                                                        nticks=self._number_y_ticks)
        ax_ticks = np.arange(0, ax_max_offset + ax_tick_size, ax_tick_size)
        ax_tick_labels = np.arange(0, (ax_max_offset / label_divider) + ax_tick_size_label, ax_tick_size_label)
        return ax_ticks, ax_tick_labels, unit

    def bar_plot(self, data, ax, colors, x_label, y_label, combinations):
        self.is_correct_data_type(data)
        ax_dataframe = pd.DataFrame.from_dict(data)
        ax_x_data = np.arange(0, len(ax_dataframe.columns), 1)
        ax_y_data = np.mean(ax_dataframe)
        ax_y_std = np.std(ax_dataframe)

        ax.bar(ax_x_data,
               ax_y_data,
               yerr=ax_y_std,
               color=colors,
               **self._barplot_kwargs, )

        sns.swarmplot(data=ax_dataframe,
                      ax=ax,
                      palette=colors,
                      **self._swarmplot_kwargs, )

        self.format_box_plot_graphics(ax, colors)
        ax.set_xticks(ax_x_data)
        ax.set_xticklabels(x_label, fontsize=self._fontsize_label, rotation=self._rotation_label)
        ax.set_xlim(ax_x_data[0] - 0.5, ax_x_data[-1] + 0.5)

        ax_y_max, ax_y_min = self.get_max_values(ax_dataframe)
        ax_y_max_offset, ax_y_min_offset = self.get_max_values_with_offset(ax_y_max, ax_y_min)
        ax_y_ticks, ax_y_tick_labels, unit = self.generate_ticks_time(ax_y_max_offset, ax_y_min_offset)
        ax.set_yticks(ax_y_ticks)
        ax.set_yticklabels(ax_y_tick_labels, fontsize=self._fontsize_label)
        y_label = y_label + " ({})".format(unit)
        ax.set_ylabel(y_label, fontsize=self._fontsize_title, labelpad=self._pad_labels)
        ax.set_ylim(ax_y_min_offset, ax_y_max_offset)

        pvals = self.generate_pvals(ax_dataframe, combinations, method=self._stat_method)
        if self._verbose:
            print("PLOT{}, pvals: {}".format("", pvals))
        self.display_significance(ax, pvals, combinations, ax_y_max, ax_y_min)

    def plot_bar_plot(self, data, colors=None, x_label="x_label",
                      y_label="y_label", figsize=(20, 10), dpi=300.,
                      combinations=None):
        plt.figure(figsize=(figsize[0] / self._cm_to_inch, figsize[1] / self._cm_to_inch),
                   dpi=dpi)
        ax0 = plt.subplot(1, 1, 1)
        self.bar_plot(data, ax0, colors, x_label, y_label, combinations)
        plt.tight_layout()
        plt.show()

    def cumulative_plot(self, data, ax, colors, x_label, y_label):
        self.is_correct_data_type(data)
        first_group = True
        ax_dataframe = pd.DataFrame.from_dict(data)
        ax_y_max = 0
        ax_y_min = 0

        for i, group in enumerate(ax_dataframe.columns):
            column_data = ax_dataframe[group]
            filtered_column_data = column_data[~column_data.isnull()]
            numpy_column_data = np.array(list(filtered_column_data), dtype=np.float)
            ax_x_data = numpy_column_data[0][:, 0]
            ax_y_data = np.mean(numpy_column_data[:, :, 1], axis=0)
            ax_y_std = np.std(numpy_column_data[:, :, 1], axis=0)
            ax_y_mean_plus_std = [i + j for i, j in zip(ax_y_data, ax_y_std)]
            ax_y_mean_min_std = [i - j for i, j in zip(ax_y_data, ax_y_std)]
            if np.max(ax_y_mean_plus_std) > ax_y_max:
                ax_y_max = np.max(ax_y_mean_plus_std)
            if np.min(ax_y_mean_min_std) < ax_y_min:
                ax_y_min = np.min(ax_y_mean_min_std)

            if first_group:
                ax_x_ticks, ax_x_tick_labels, unit = self.generate_ticks_time(ax_x_data[-1], ax_x_data[0])
                ax.set_xticks(ax_x_ticks)
                ax.set_xticklabels(ax_x_tick_labels, fontsize=self._fontsize_label)
                x_label = x_label + " ({})".format(unit)
                ax.set_xlabel(x_label, fontsize=self._fontsize_title, labelpad=self._pad_labels)
                ax.set_xlim(ax_x_data[0], ax_x_data[-1])
                first_group = False

            ax.plot(ax_x_data, ax_y_data, color=colors[i], label=group, **self._cumulativeplot_meanline_kwargs)
            ax.plot(ax_x_data, ax_y_mean_plus_std, color=colors[i], **self._cumulativeplot_stdline_kwargs)
            ax.plot(ax_x_data, ax_y_mean_min_std, color=colors[i], **self._cumulativeplot_stdline_kwargs)
            ax.fill_between(ax_x_data, ax_y_data, ax_y_mean_plus_std, alpha=0.1, color=colors[i])
            ax.fill_between(ax_x_data, ax_y_data, ax_y_mean_min_std, alpha=0.1, color=colors[i])

        self.format_figure_canvas(ax)

        ax_y_max_offset, ax_y_min_offset = self.get_max_values_with_offset(ax_y_max, ax_y_min)
        ax_y_ticks, ax_y_tick_labels, unit = self.generate_ticks_time(ax_y_max_offset, ax_y_min_offset)
        ax.set_yticks(ax_y_ticks)
        ax.set_yticklabels(ax_y_tick_labels, fontsize=self._fontsize_label)
        y_label = y_label + " ({})".format(unit)
        ax.set_ylabel(y_label, fontsize=self._fontsize_title, labelpad=self._pad_labels)
        ax.set_ylim(ax_y_min_offset, ax_y_max_offset)

    def plot_cumulative_plot(self, data, colors=None, x_label="x_label",
                             y_label="y_label", figsize=(20, 10), dpi=300.):
        plt.figure(figsize=(figsize[0] / self._cm_to_inch, figsize[1] / self._cm_to_inch),
                   dpi=dpi)
        ax0 = plt.subplot(1, 1, 1)
        self.cumulative_plot(data, ax0, colors, x_label, y_label)
        plt.tight_layout()
        plt.show()

    def plot_complete_plot(self, ax0_data, ax1_data, ax2_data, colors=None, ax0_label="x_label",
                           ay0_label="y_label", ax1_label="x_label", ay1_label="y_label",
                           ax2_label="x_label", ay2_label="y_label", figsize=(20, 10), dpi=300.,
                           combinations=None):
        fig = plt.figure(figsize=(figsize[0] / self._cm_to_inch, figsize[1] / self._cm_to_inch),
                         dpi=dpi)
        gs = gridspec.GridSpec(ncols=4, nrows=1, figure=fig)

        ax0 = fig.add_subplot(gs[0, 0])
        self.bar_plot(ax0_data, ax0, colors, ax0_label, ay0_label, combinations)

        ax1 = fig.add_subplot(gs[0, 1:3])
        self.cumulative_plot(ax1_data, ax1, colors, ax1_label, ay1_label)

        ax2 = fig.add_subplot(gs[0, 3])
        self.bar_plot(ax2_data, ax2, colors, ax2_label, ay2_label, combinations)

        plt.tight_layout()
        plt.show()

    def get_color_for_behavior(self, behavior):
        keys = np.array(list(self._behavior_color_map.keys()), dtype=str)
        beh_mask = [behavior.startswith(i) for i in keys]
        if any(beh_mask):
            behavior_in_dict = keys[beh_mask][0]
            return self._behavior_color_map[behavior_in_dict]
        else:
            return None

    def redefine_behavior_color_map(self, behavior_color_map):
        self._behavior_color_map = behavior_color_map

    def complete_raster_plot(self, ax, data, behaviors):
        for behavior_name in behaviors:
            # for behavior_name, behavior_data in data.items():
            behavior_color = self.get_color_for_behavior(behavior_name)
            if not behavior_color is None:
                for s, e in data[behavior_name]:
                    rect = Rectangle((s, 0),
                                     e - s,
                                     1,
                                     linewidth=0,
                                     edgecolor='black',
                                     facecolor=behavior_color,
                                     label=behavior_name)
                    ax.add_patch(rect)
            else:
                for s, e in data[behavior_name]:
                    rect = Rectangle((s, 0),
                                     e - s,
                                     1,
                                     linewidth=0,
                                     edgecolor='black',
                                     label=behavior_name)
                    ax.add_patch(rect)

        self.format_figure_canvas(ax)
        ax.set_yticks([])
        ax.set_ylim(0, 1)

    def generate_legend_data_behaviors(self, behaviors):
        legend_handles = []
        legend_labels = []
        for b in behaviors:
            color = self.get_color_for_behavior(b)
            if color is not None:
                rec = Rectangle((0, 0),
                                1,
                                1,
                                linewidth=0,
                                edgecolor='black',
                                facecolor=color, )
            else:
                rec = Rectangle((0, 0),
                                1,
                                1,
                                linewidth=0,
                                edgecolor='black', )
            legend_handles.append(rec)
            legend_labels.append(b)
        return legend_handles, legend_labels

    def plot_raster_plot(self, data, behaviors, figsize=(50, 30), dpi=300., limit=3600,
                         x_label="x_label"):
        is_group = any(isinstance(i, dict) for i in data.values())
        legend_handles, legend_labels = self.generate_legend_data_behaviors(behaviors)

        plt.figure(figsize=(figsize[0] / self._cm_to_inch, figsize[1] / self._cm_to_inch), dpi=dpi)
        plot_n = 1
        if is_group:
            n_animals = len(data.keys())
            for animal_name, animal_data in data.items():
                ax = plt.subplot(n_animals, 1, plot_n)
                if plot_n == 1:
                    ax.legend(handles=legend_handles,
                              labels=legend_labels,
                              loc=2,
                              fontsize=2.,
                              ncol=len(legend_labels),
                              bbox_to_anchor=(0, 1.7))
                if plot_n == n_animals:
                    ax_x_ticks, ax_x_tick_labels, unit = self.generate_ticks_time(limit, 0)
                    ax.set_xticks(ax_x_ticks)
                    ax.set_xticklabels(ax_x_tick_labels, fontsize=self._fontsize_label)
                    x_label = x_label + " ({})".format(unit)
                    ax.set_xlabel(x_label, fontsize=self._fontsize_title, labelpad=self._pad_labels)
                    ax.set_xlim(0, limit)
                else:
                    ax.set_xticks([])
                    ax.set_xlim(0, limit)
                plot_n += 1
                self.complete_raster_plot(ax, animal_data, behaviors)
        else:
            ax = plt.subplot(1, 1, 1)
            self.complete_raster_plot(ax, data, behaviors)
            ax_x_ticks, ax_x_tick_labels, unit = self.generate_ticks_time(limit, 0)
            ax.set_xticks(ax_x_ticks)
            ax.set_xticklabels(ax_x_tick_labels, fontsize=self._fontsize_label)
            x_label = x_label + " ({})".format(unit)
            ax.set_xlabel(x_label, fontsize=self._fontsize_title, labelpad=self._pad_labels)
            ax.set_xlim(0, limit)
        plt.tight_layout()
        plt.show()


# class ManipulateBehavioralData():
#     def __init__(self):
#         pass


class BehavioralDataAnimal(Plotting):
    def __init__(self, animal_file_path, transpose=True):
        super().__init__()
        self._animal_file_path = animal_file_path
        self._animal_name = os.path.basename(self._animal_file_path).split(".")[0]
        self._dataframe_animal = self.load_behavior_as_dataframe(transpose=transpose)
        self._time_add_cotton = float(self.check_and_get_column_data(self._dataframe_animal,
                                                                     "tcotton").iloc[0])
        self._general_initiation_time = float(self.check_and_get_column_data(self._dataframe_animal,
                                                                             "tinitiate").iloc[0])
        self._time_video_ends = float(self.check_and_get_column_data(self._dataframe_animal,
                                                                     "videoend").iloc[0])
        self._behavior_names = self.get_behaviors_from_dataframe()
        self._behavior_data = self.get_complete_behavior_data_from_source()

    def load_behavior_as_dataframe(self, transpose=True):
        """
        Method to load excel behavioral spreadsheets as a dataframe

        :param bool transpose: Transpose data if the behavior excel spreadsheet is written in horizontal
        :return: The loaded dataframe with behavioral data
        """
        df = pd.read_excel(self._animal_file_path, header=None)
        if transpose:
            df = df.T
            labels_first_row = df.iloc[0]
            df = df.drop(0)
            df.columns = labels_first_row
        df.columns = df.columns.str.lower()
        return df

    def get_behaviors_from_dataframe(self):
        """
        Method to enumerate all the behaviors segmented for an animal

        :return: The names of the different behaviors segmented for this animal
        """
        keys = self._dataframe_animal.keys()
        behavioral_names = [i.split("tstart")[-1] for i in keys \
                            if i.startswith("tstart")]
        return behavioral_names

    def extract_and_clean_column_data(self, key):
        """
        Method to extract data based on the name of the column in a dataframe

        :param key: Name of the column to be extracted
        :return dataframe: The data of the column
        """
        column_data = self.check_and_get_column_data(self._dataframe_animal, key, animal_name=self._animal_name)
        column_data = self.remove_NaNs_from_df(column_data)
        return column_data

    def get_single_behavior_data_from_source(self, behavior):
        """
        Method to extract the data of a single behavior for an animal

        :param str behavior: The behavior to be extracted
        :return: The data for the specified behavior
        """
        beh = str(behavior).lower()
        start_label = "tstart{}".format(beh)
        start_data = self.extract_and_clean_column_data(start_label)
        end_label = "tend{}".format(beh)
        end_data = self.extract_and_clean_column_data(end_label)
        data = np.column_stack((start_data, end_data))
        self._check_problems_behavior_data(data, beh)
        return data

    def get_complete_behavior_data_from_source(self):
        """
        Method to extract the data of all behaviors for an animal

        :return: The behavioral data in the form of a dictionary
        """
        behavioral_data = {}
        for b in self._behavior_names:
            data = self.get_single_behavior_data_from_source(b)
            behavioral_data[b] = data
        return behavioral_data

    def get_single_behavior_data(self, behavior, time_limit=3600, init_type=None, behavior_init=None):
        """
        Method to extract the data of a single behavior from within the class instance

        :param str behavior: The behavior to be extracted
        :return: The data for the specified behavior
        :param int time_limit: The time limit of data to be extracted
        :param string init_type: The type of initiation to be used
        :param string behavior_init: If init_type is 'specific': the behavior to be used to compute initiation
        """
        behavior_data = self.check_and_get_dict_entry(self._behavior_data, behavior,
                                                      animal_name=self._animal_name,
                                                      available_beh=self._behavior_names)
        time_initiate = self.get_initiation_from_condition(init_type, behavior=behavior_init)
        filtered_start_data, filtered_end_data = self.filter_start_end_bouts_in_time(behavior_data, time_initiate, time_limit)
        behavior_data = np.column_stack((filtered_start_data, filtered_end_data))
        return behavior_data

    def get_complete_behavior_data(self, time_limit=3600, init_type=None, behavior_init=None):
        """
        Method to extract the data of all behaviors from within the class instance

        :param int time_limit: The time limit of data to be extracted
        :param string init_type: The type of initiation to be used
        :param string behavior_init: If init_type is 'specific': the behavior to be used to compute initiation
        :return: The data for the all the behaviors
        """
        complete_behavior_data = {}
        time_initiate = self.get_initiation_from_condition(init_type, behavior=behavior_init)
        for behavior_name, behavior_data in self._behavior_data.items():
            filtered_start_data, filtered_end_data = self.filter_start_end_bouts_in_time(behavior_data, time_initiate,
                                                                                         time_limit)
            behavior_data = np.column_stack((filtered_start_data, filtered_end_data))
            complete_behavior_data[behavior_name] = behavior_data
        return complete_behavior_data

    def combine_behaviors(self, behaviors_to_combine, sink_dict_key):
        """
        Method to combine behaviors together and create a new key entry in self._behavior_data

        :param list behaviors_to_combine: List of names of behaviors to be combined
        :param str sink_dict_key: The key entry for the combined behavior in self._behavior_data
        :return:
        """
        data = np.concatenate([self.get_single_behavior_data(b) for b in behaviors_to_combine])
        arg_sort_start = np.argsort(data[:, 0])
        data_combined = data[arg_sort_start]
        self._check_problems_behavior_data(data_combined, sink_dict_key)
        self._behavior_data[sink_dict_key] = data_combined
        self._behavior_names.append(sink_dict_key)

    def get_general_initiation_data_for_animal(self):
        """
        Method to get the general initiation time (labeled as "tinitiate" in the spreadsheet) of the animal
        If the entry is missing returns 0
        :return: The general initiation time of the animal
        """
        return self._general_initiation_time

    def get_specific_initiation_data_for_animal(self, behavior):
        """
        Method to get the specific initiation time of the animal for a given behavior

        :param behavior: The behavior for which the initiation time is calculated
        :return: The initiation time or 0 if the behavior is empty or missing
        """
        try:
            data = self.get_single_behavior_data(behavior)
            return data[0, 0]
        except:
            print("No {} bouts were found for animal:{}"
                  " using start of video for initiation".format(behavior, self._animal_name))
            return 0

    def get_initiation_from_condition(self, behavior_init, behavior=None):
        """
        Method to get the initiation time depending on the chosen method

        :param behavior_init: The type of initiation to be extracted
        :param behavior: If behavior_init is "specific" this keyword argument corresponds to the behavior to be
        used for initiation
        :return:
        """
        if behavior_init == "specific":
            initiation_time = self.get_specific_initiation_data_for_animal(behavior)
        elif behavior_init == "general":
            initiation_time = self.get_general_initiation_data_for_animal()
        else:
            initiation_time = 0
        return initiation_time

    def get_cumulative_data_animal(self, behavior, time_limit=3600, behavior_init=None):
        """
        Method to extract cumulative data from a behavior

        :param behavior: The behavior from which the cumulative data has to be extracted
        :param time_limit: The time limit of data to be extracted
        :param behavior_init: The type of initiation to be used
        :return: The cumulative data
        """
        behavioral_data = self.get_single_behavior_data(behavior)
        time_initiate = self.get_initiation_from_condition(behavior_init, behavior=behavior)
        return self.transform_start_end_to_cumulative(behavioral_data, time_initiate, time_limit)

    def transform_start_end_to_cumulative(self, behavioral_data, time_initiate, time_limit):
        """
        Method to transform start end type of data into cumulative data

        :param behavioral_data: the behavioral data in the (start, end) form
        :param time_initiate: The time at which the behavior is initiated
        :param time_limit: The time limit of data to be extracted
        :return: The cumulative data
        """
        time_data_no_gap = np.arange(0, time_limit + 1, 1)
        cumulative_time_data_no_gap = np.zeros(time_limit + 1)

        filtered_start_data, filtered_end_data = self.filter_start_end_bouts_in_time(behavioral_data,
                                                                                     time_initiate,
                                                                                     time_limit)
        filtered_data = np.column_stack((filtered_start_data, filtered_end_data))
        length_bouts = np.squeeze(np.diff(filtered_data))
        cumulative_time_data = np.cumsum(length_bouts)

        for i in range(len(filtered_start_data)):
            if i < len(filtered_start_data) - 1:
                diff_0 = int(filtered_end_data[i]) - int(filtered_start_data[i])
                cumulative_time_data_no_gap[int(filtered_start_data[i]): \
                                            int(filtered_end_data[i] + 1)] \
                    = np.arange(cumulative_time_data[i] - diff_0,
                                cumulative_time_data[i] + 1)
                cumulative_time_data_no_gap[int(filtered_end_data[i]):
                                            int(filtered_start_data[i + 1] + 1)] \
                    = cumulative_time_data[i]
            else:
                diff_0 = int(filtered_end_data[i]) - int(filtered_start_data[i])
                cumulative_time_data_no_gap[int(filtered_start_data[i]): \
                                            int(filtered_end_data[i] + 1)] \
                    = np.arange(cumulative_time_data[i] - diff_0,
                                cumulative_time_data[i] + 1)
                cumulative_time_data_no_gap[int(filtered_end_data[i]): \
                                            int(time_limit + 1)] \
                    = cumulative_time_data[i]

        behavioral_data = np.column_stack((time_data_no_gap, cumulative_time_data_no_gap))
        return behavioral_data

    def filter_start_end_bouts_in_time(self, behavioral_data, time_initiate, time_limit):
        """
        Method to filter the (start, end) behavioral data. This clears any data:
        - Before the initiation time
        - After the time limit

        :param behavioral_data: The (start, end) behavioral data
        :param time_initiate: The time at which the behavior is initiated
        :param time_limit: The time limit of data to be extracted
        :return: The filtered (start, end) behavioral data
        """
        start_data_raw, end_data_raw = self.filter_bouts_before_initiation(behavioral_data, time_initiate)
        if start_data_raw.size == 0 or end_data_raw.size == 0:
            return np.array([]), np.array([])
        start_data, end_data = self.filter_bouts_after_time_limit(start_data_raw, end_data_raw, time_initiate,
                                                                  time_limit)
        if start_data.size == 0 or end_data.size == 0:
            return np.array([]), np.array([])
        filtered_data = np.column_stack((start_data, end_data))
        filtered_start_data = filtered_data[:, 0]
        filtered_end_data = filtered_data[:, 1]

        filtered_start_data, filtered_end_data = self.substract_initiation_time(filtered_start_data,
                                                                                filtered_end_data,
                                                                                time_initiate)

        filtered_start_data, filtered_end_data = self.insert_zero_and_limit_time_to_data(filtered_start_data,
                                                                                         filtered_end_data,
                                                                                         time_limit)
        return filtered_start_data, filtered_end_data

    def _check_ordering_behavior_data(self, behavioral_data, behavior):
        """
        Method that checks if the ordering of bouts of a given behavior in time
        is correct

        :param behavioral_data: The behavioral data to be checked
        :param behavior: The tested behavior
        :return:
        """
        x_shape, y_shape = behavioral_data.shape
        target_order_values = np.arange(0, x_shape, 1)
        argsort_start = np.argsort(behavioral_data[:, 0])
        if not (argsort_start == target_order_values).all():
            raise BehaviorException("ordering of start data is wrong in animal:" \
                                    "{} for behavior:{}" \
                                    .format(self._animal_name, behavior))
        argsort_end = np.argsort(behavioral_data[:, 1])
        if not (argsort_end == target_order_values).all():
            raise BehaviorException("ordering of end data is wrong in animal:" \
                                    "{} for behavior:{}" \
                                    .format(self._animal_name, behavior))

    def _check_overlapping_behavior_data(self, behavioral_data, behavior):
        """
        Method that checks if the bouts of a given behavior overlap in time

        :param behavioral_data: The behavioral data to be checked
        :param behavior: The tested behavior
        :return:
        """
        shifted_data = np.column_stack((behavioral_data[1:, 0], behavioral_data[:-1, 1]))
        diff = np.diff(shifted_data)
        if not all(diff <= 0):
            raise BehaviorException("some behavioral bouts overlap in animal:{} for behavior:{}" \
                                    .format(self._animal_name, behavior))

    def _check_problems_behavior_data(self, behavioral_data, behavior):
        """
        Method that check simultaneously for ordering and overlapping problems in
        behavioral data

        :param behavioral_data: The behavioral data to be checked
        :param behavior: The tested behavior
        :return:
        """
        self._check_ordering_behavior_data(behavioral_data, behavior)
        self._check_overlapping_behavior_data(behavioral_data, behavior)

    @staticmethod
    def check_and_get_column_data(df, key, animal_name=""):
        """
        Static method to check if a column name exists in a dataframe and returns it

        :param df: The dataframe from which the column has to be extracted
        :param key: Name of the column to be extracted
        :param animal_name: [OPTIONAL] the name of the animal for debugging
        :return: The data of the column
        """
        if key in df.columns:
            return df[key]
        else:
            if animal_name:
                raise BehaviorException("no entry {} for animal:{}".format(key, animal_name))
            else:
                raise BehaviorException("no entry {} in df".format(key))

    @staticmethod
    def check_and_get_dict_entry(dictionary, key, animal_name="", available_beh=""):
        """
        Static method to check if a key name exists in a dictionary and returns it

        :param dict dictionary: The dictionary from which the entry has to be extracted
        :param key: Name of the key to be extracted
        :param string animal_name: [OPTIONAL] the name of the animal for debugging
        :param list/string available_beh: [OPTIONAL] all the available behaviors for debugging
        :return: The data of the key in the dictionary
        """
        if key in dictionary.keys():
            return dictionary[key]
        else:
            if animal_name:
                raise BehaviorException("no entry {} for animal: {}."
                                        "Available behaviors are: {}".format(key, animal_name, available_beh))
            else:
                raise BehaviorException("no entry {} in df".format(key))

    @staticmethod
    def filter_bouts_before_initiation(behavioral_data, time_initiate):
        """
        Static method to filter bouts occurring before a certain time

        :param behavioral_data: The behavioral data to be filtered
        :param time_initiate: The initiation time
        :return: Filtered (start, end) data
        """
        if behavioral_data.size == 0:
            return np.array([]), np.array([])
        initiate_mask = behavioral_data[:, 1] >= time_initiate
        start_data_raw = behavioral_data[:, 0][initiate_mask]
        end_data_raw = behavioral_data[:, 1][initiate_mask]
        if start_data_raw.size == 0 or end_data_raw.size == 0:
            return np.array([]), np.array([])
        if start_data_raw[0] < time_initiate:
            start_data_raw[0] = time_initiate
        return start_data_raw, end_data_raw

    @staticmethod
    def filter_bouts_after_time_limit(start_data, end_data, time_initiate, time_limit):
        """
        Static method to filter bouts occurring after the time limit

        :param start_data: The start behavioral data to be filtered
        :param end_data: The end behavioral data to be filtered
        :param time_initiate: The initiation time
        :param time_limit: The time limit
        :return: Filtered (start, end) data
        """
        start_mask = start_data <= time_limit + time_initiate
        start_data = start_data[start_mask]
        end_data = end_data[start_mask]
        if start_data.size == 0 or end_data.size == 0:
            return np.array([]), np.array([])
        if end_data[-1] > time_limit + time_initiate:
            end_data[-1] = time_limit + time_initiate
        return start_data, end_data

    @staticmethod
    def substract_initiation_time(start_data, end_data, initiation_time):
        """
        Static method that subtracts the initiation time to (start, end) data
        (to put it back to zero)

        :param start_data: The start behavioral data
        :param end_data: The end behavioral data
        :param initiation_time: The initiation time
        :return: The subtracted (start, end) data
        """
        start_data = start_data - initiation_time
        end_data = end_data - initiation_time
        return start_data, end_data

    @staticmethod
    def insert_zero_and_limit_time_to_data(start_data, end_data, time_limit):
        if start_data[0] != 0:
            start_data = np.insert(start_data, 0, 0)
            end_data = np.insert(end_data, 0, 0)
        if end_data[-1] != time_limit:
            start_data = np.append(start_data, time_limit)
            end_data = np.append(end_data, time_limit)
        return start_data, end_data

    @staticmethod
    def remove_NaNs_from_df(df):
        """
        Method to remove NaNs from a dataframe

        :param df: The dataframe to be cleaned
        :return: The cleaned dataframe
        """
        return df[~pd.isnull(df)]


class BehavioralDataGroup(Plotting):
    def __init__(self, working_directory, group):
        super(BehavioralDataGroup, self).__init__()
        self._animal_names = [str(i) for i in group]
        self._animal_file_paths = []
        self._animal_data = {}
        for file_name in natsorted(os.listdir(working_directory)):
            if file_name.endswith(".xlsx") and not file_name.startswith("._"):
                animal_name = file_name.split(".")[0]
                if animal_name in self._animal_names:
                    file_path = os.path.join(working_directory, file_name)
                    self._animal_file_paths.append(file_path)
                    self._animal_data[animal_name] = BehavioralDataAnimal(file_path)
        self._behavior_names = self.get_behavior_names_group()

    def combine_behaviors_for_group(self, behaviors_to_combine, sink_dict_key):
        for animal_data in self._animal_data.values():
            animal_data.combine_behaviors(behaviors_to_combine, sink_dict_key)

    def get_complete_behavior_data_for_group(self, time_limit=3600, init_type=None, behavior_init=None):
        behavioral_data = {}
        for animal_name, animal_data in self._animal_data.items():
            behavioral_data[animal_name] = animal_data.get_complete_behavior_data(time_limit=time_limit,
                                                                                  init_type=init_type,
                                                                                  behavior_init=behavior_init)
        return behavioral_data

    def get_single_behavioral_data_for_group(self, behavior, time_limit=3600, init_type=None, behavior_init=None):
        behavioral_data = {}
        for animal_name, animal_data in self._animal_data.items():
            behavioral_data[animal_name] = animal_data.get_single_behavior_data(behavior,
                                                                                time_limit=time_limit,
                                                                                init_type=init_type,
                                                                                behavior_init=behavior_init)
        return behavioral_data

    def get_general_initiation_data_for_group(self):
        general_initiation_data = {}
        for animal_name, animal_data in self._animal_data.items():
            general_initiation_data[animal_name] = animal_data.get_general_initiation_data_for_animal()
        return general_initiation_data

    def get_specific_initiation_data_for_group(self, behavior):
        specific_initiation_data = {}
        for animal_name, animal_data in self._animal_data.items():
            specific_initiation_data[animal_name] = animal_data.get_specific_initiation_data_for_animal(behavior)
        return specific_initiation_data

    def get_cumulative_data_group(self, behavior, time_limit=3600, behavior_init=None):
        cumulative_data = {}
        for animal_name, animal_data in self._animal_data.items():
            cumulative_data[animal_name] = animal_data.get_cumulative_data_animal(behavior,
                                                                                  time_limit=time_limit,
                                                                                  behavior_init=behavior_init)
        return cumulative_data

    def get_behavior_names_group(self):
        behaviors = []
        for animal_name, animal_data in self._animal_data.items():
            for beh in animal_data._behavior_names:
                if beh not in behaviors:
                    behaviors.append(beh)
        return behaviors


class BehavioralDataGroups(Plotting):
    def __init__(self, working_directory, all_groups, groups):
        super().__init__()
        self._group_names = [str(i) for i in groups]
        self._group_data = {}
        for g in groups:
            self._group_data[g] = BehavioralDataGroup(working_directory, all_groups[g])
        self._behavior_names = self.get_behavior_names_groups()

    def combine_behaviors_for_groups(self, behaviors_to_combine, sink_dict_key):
        for group_data in self._group_data.values():
            group_data.combine_behaviors_for_group(behaviors_to_combine, sink_dict_key)
        if not sink_dict_key in self._behavior_names:
            self._behavior_names.append(sink_dict_key)
        print("\nAdded {} entry to _behavioral_data attr".format(sink_dict_key))

    def get_complete_behavior_data_for_groups(self, time_limit=3600, init_type=None, behavior_init=None):
        behavioral_data = {}
        for group_name, group_data in self._group_data.items():
            behavioral_data[group_name] = group_data.get_complete_behavior_data_for_group(time_limit=time_limit,
                                                                                          init_type=init_type,
                                                                                          behavior_init=behavior_init)
        return behavioral_data

    def get_single_behavior_data_for_groups(self, behavior, time_limit=3600, init_type=None, behavior_init=None):
        behavioral_data = {}
        for group_name, group_data in self._group_data.items():
            behavioral_data[group_name] = group_data.get_single_behavioral_data_for_group(behavior,
                                                                                          time_limit=time_limit,
                                                                                          init_type=init_type,
                                                                                          behavior_init=behavior_init)
        return behavioral_data

    def get_general_initiation_data_groups(self):
        general_initiation_data = {}
        for group_name, group_data in self._group_data.items():
            general_initiation_data[group_name] = group_data.get_general_initiation_data_for_group()
        return general_initiation_data

    def get_specific_initiation_data_groups(self, behavior):
        specific_initiation_data = {}
        for group_name, group_data in self._group_data.items():
            specific_initiation_data[group_name] = group_data.get_specific_initiation_data_for_group(behavior)
        return specific_initiation_data

    def get_cumulative_data_groups(self, behavior, time_limit=3600, behavior_init=None):
        cumulative_data = {}
        for group_name, group_data in self._group_data.items():
            cumulative_data[group_name] = group_data.get_cumulative_data_group(behavior,
                                                                               time_limit=3600,
                                                                               behavior_init=behavior_init)
        return cumulative_data

    def get_all_start_end_data_groups(self, init_behavior, time_limit=3600, behavior_init=None):
        default = False
        behavioral_data = self.get_complete_behavior_data_for_groups()
        if behavior_init == "specific":
            initiation_data = self.get_specific_initiation_data_groups(init_behavior)
        elif behavior_init == "general":
            initiation_data = self.get_general_initiation_data_groups()
        else:
            default = True
            initiation_time = 0
        start_end_data = {}
        for group_name, group_data in behavioral_data.items():
            animal_start_end_data = {}
            for animal_name, animal_data in group_data.items():
                if not default:
                    initiation_time = initiation_data[group_name][animal_name]
                behavior_start_end_data = {}
                for behavior_name, behavior_data in animal_data.items():
                    behavior_start_end_data[behavior_name] = self.filter_start_end(behavior_data, initiation_time,
                                                                                   time_limit)
                animal_start_end_data[animal_name] = behavior_start_end_data
            start_end_data[group_name] = animal_start_end_data
        return start_end_data

    # def get_all_start_end_data_groups(self, init_behavior, time_limit=3600, behavior_init=None):
    #     start_end_data = {}
    #     for group_name, group_data in self._group_data.items():
    #         start_end_data[group_name] = group_data.get_all_start_end_data_group(init_behavior,
    #                                                                              time_limit=time_limit,
    #                                                                              behavior_init=behavior_init)
    #     return start_end_data
    #
    # def get_specific_start_end_data_groups(self, init_behavior, time_limit=3600, behavior_init=None):
    #     start_end_data = {}
    #     for group_name, group_data in self._group_data.items():
    #         start_end_data[group_name] = group_data.get_specific_start_end_data_group(init_behavior,
    #                                                                                   time_limit=time_limit,
    #                                                                                   behavior_init=behavior_init)
    #     return start_end_data

    def get_behavior_names_groups(self):
        behaviors = []
        for group_name, group_data in self._group_data.items():
            beh = group_data.get_behavior_names_group()
            for b in beh:
                if b not in behaviors:
                    behaviors.append(b)
        return behaviors

    @staticmethod
    def get_total_time_behaving_from_cumulative_data(cumulative_dict):
        sink = {}
        for group_name, group_data in cumulative_dict.items():
            animal_dict = {}
            for animal_name, animal_data in group_data.items():
                animal_dict[animal_name] = animal_data[-1, -1]
            sink[group_name] = animal_dict
        return sink
