import os
import behavior_ploting as bp

all_groups = {"fPregnant": [6,9,13,14,24,28],
              "fPseudo": [4,10,11,15,23,27],
              "fVirgin": [2,5,8,12,36,37],
              "mMated": [57,58,59,60,61,62,63],
              "mVirgin": [45,47,48,49],
              "fVirginPup": [101, 102, 103, 104, 105, 106],}

metadata_groups = {"fPregnant": ["#f97c7c", "Pregnant \u2640"],
                   "fPseudo": ["#ffb347", "Pseudo \u2640"],
                   "fVirgin": ["#84d9ff", "Virgin \u2640"],
                   "mMated": ["#77dd77", "Mated \u2642"],
                   "mVirgin": ["#03c03c", "Virgin \u2642"],
                   "fVirginPup": ["#779ecb", "Virgin \u2640 + pup"]}

working_directory = "/network/lustre/dtlake01/renier/Thomas/thomas.topilko/Experiments"\
                    "/Nesting_Project/Behavior/Virgin_Pregnant_Activity_Mapping_Females"\
                    "/170424/plot_cumulative_behavior/data"
print(os.path.exists(working_directory))

group_names = ("fVirgin", "fPseudo", "fPregnant", "mMated")
group_colors = [metadata_groups[i][0] for i in group_names]
group_labels = [metadata_groups[i][1] for i in group_names]

behavior_data_groups = bp.BehavioralDataGroups(working_directory,
                                               all_groups,
                                               group_names)

behavior_to_test = "nesting"
behavioral_label = "nesting"
ax0_data = behavior_data_groups.get_general_initiation_data_groups()
ax1_data = behavior_data_groups.get_cumulative_data_groups(behavior_to_test,
                                                           time_limit=3600,
                                                           behavior_init=None)
ax2_data = behavior_data_groups.get_total_time_behaving_from_cumulative_data(ax1_data)

# figsize = (30, 30)
# y_label = "Time spent {}".format(behavioral_label)
# behavior_data_groups.plot_box_plot(ax0_data,
#                                    colors=group_colors,
#                                    x_label=group_labels,
#                                    y_label=y_label,
#                                    figsize=figsize,
#                                    dpi=300.)

# x_label = "Time"
# y_label = "Time spent {}".format(behavioral_label)
# behavior_data_groups.plot_cumulative_plot(ax1_data,
#                                      colors=group_colors,
#                                      x_label=x_label,
#                                      y_label=y_label,
#                                      figsize=figsize,
#                                      dpi=300.)


# plot_kwargs = {"colors": group_colors,
#                "ax0_label": group_labels,
#                "ay0_label": "Delay to initiate {}".format(behavioral_label),
#                "ax1_label": "Time",
#                "ay1_label": "Time spent {}".format(behavioral_label),
#                "ax2_label": group_labels,
#                "ay2_label": "Time spent {}".format(behavioral_label),
#                "figsize": (100, 50),
#                "dpi": 700.,
#                "combinations": [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3], [1, 3]],
#                }
#
# behavior_data_groups.plot_complete_plot(ax0_data, ax1_data, ax2_data, **plot_kwargs)

test_data = behavior_data_groups.get_all_start_end_data_groups(None,
                                                               time_limit=1*3600,
                                                               behavior_init="general")

test_data.combine_behaviors_for_groups(["nesting", "dig"], "global_nesting")
test_data.combine_behaviors_for_groups(["drink", 'snifffood'], "other")
print(test_data._behavior_names)

behavior_color_map = {"global_nesting": "#f97c7c",
                      "sleep": "#84d9ff",
                      "other": "#77dd77",
                      "groom": "#f49ac2"}
behavior_data_groups.redefine_behavior_color_map(behavior_color_map)

plot_kwargs = {"figsize": (50, 30),
               "dpi": 300.,
               "x_label": "Time",
               }

behavior_data_groups.plot_raster_plot(test_data["fVirgin"],
                                      behavior_color_map.keys(),
                                      **plot_kwargs)


file_path = "/network/lustre/dtlake01/renier/Thomas/thomas.topilko/Experiments/Nesting_Project/Behavior/" \
            "Virgin_Pregnant_Activity_Mapping_Females/170424/plot_cumulative_behavior/data/2.xlsx"
test = bp.BehavioralDataAnimal(file_path,
                               transpose=True,)

print(test._animal_name)
print(test._behavior_names)
test.combine_behaviors(["nesting", "dig"], "global_nesting")
print(test._behavior_names)
nesting_data = test.get_single_behavior_data("global_nesting")
behavior_data = test.get_complete_behavior_data()
cumulative_test = test.get_cumulative_data_animal("global_nesting")
import matplotlib.pyplot as plt
plt.plot(cumulative_test[:, 0], cumulative_test[:, 1])
plt.show()













