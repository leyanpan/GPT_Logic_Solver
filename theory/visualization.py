import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Visualization:
    def __init__(self):
        self.data = pd.DataFrame()

    def add_array(self, name, array: np.ndarray):
        if array.ndim == 1:
            self.data[name] = array
        else:
            for i in range(array.shape[1]):
                self.data[name + f'[{i}]'] = array[:, i]

    def display(self, start_index=0, max_len=None):
        if max_len is None or start_index + max_len > self.data.shape[0]:
            max_len = self.data.shape[0] - start_index
        data_to_display = self.data.iloc[start_index:start_index + max_len].T

        # Prepare cell colors
        cell_text = data_to_display.values.tolist()
        cell_colors = []

        for row in data_to_display.values:
            row_colors = []
            for val in row:
                if isinstance(val, bool):
                    color = 'lightgreen' if val else 'lightcoral'
                else:
                    color = 'white'
                row_colors.append(color)
            cell_colors.append(row_colors)

        fig, ax = plt.subplots(figsize=(12, len(data_to_display) // 2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=cell_text,
                         cellColours=cell_colors,
                         rowLabels=data_to_display.index,
                         colLabels=data_to_display.columns,
                         loc='center',
                         cellLoc='center',
                         colLoc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        plt.show()
