from typing import Any, List, Tuple, Union
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Figure
from numpy import ndarray
from utils.const import PARAMETRIC_MAP, TEST_PREDICTION


def make_imgs(img: ndarray, imin: Any = None, imax: Any = None) -> ndarray:
    """Apply a 3D binary mask to a 1-channel, 3D ndarray `img` by creating a 3-channel
    image with masked regions shown in transparent blue."""
    imin = img.min() if imin is None else imin
    imax = img.max() if imax is None else imax
    scaled = np.array(((img - imin) / (imax - imin)) * 255, dtype=int)  # img
    # scaled = np.array(img * 255, dtype=int)
    # scaled = img
    return scaled, imin, imax

class BrainSlices:
    def __init__(
        self,
        ratio_ec_data: ndarray,
        ratio_diff_ec_data: ndarray,
        ratio_sign_ec_data: ndarray,
        ratio_diff_sign_ec_data: ndarray,
        inter_union_vox_ec_data: ndarray,
        inter_union_distance_ec_data: ndarray,
        subject: int,
    ):
        self.fig_data_1_img, self.imin1, self.imax1 = make_imgs(ratio_ec_data)
        self.fig_data_2_img, self.imin2, self.imax2 = make_imgs(ratio_diff_ec_data)
        self.fig_data_3_img, self.imin3, self.imax3 = make_imgs(ratio_sign_ec_data)
        self.fig_data_4_img, self.imin4, self.imax4 = make_imgs(ratio_diff_sign_ec_data)
        self.fig_data_5_img, self.imin5, self.imax5 = make_imgs(inter_union_vox_ec_data)
        self.fig_data_6_img, self.imin6, self.imax6 = make_imgs(inter_union_distance_ec_data)
        self.subject: int = subject

        si, sj, sk = 128, 128, 128
        i = si // 2
        j = sj // 2
        k = sk // 2

        self.slices = [ self.get_slice(self.fig_data_1_img, i, j, k), 
                        self.get_slice(self.fig_data_2_img, i, j, k),
                        self.get_slice(self.fig_data_3_img, i, j, k), 
                        self.get_slice(self.fig_data_4_img, i, j, k),
                        self.get_slice(self.fig_data_5_img, i, j, k), 
                        self.get_slice(self.fig_data_6_img, i, j, k), 
                        ]


        self.title = ["Ratio",
        "Ratio-diff",
        "Ratio-signed",
        "Ratio-diff-signed",
        "Intersection-union-voxel",
        "Intersection-union-distance"]
        # 

    def get_slice(self, input: ndarray, i: int, j: int, k: int) -> List[Tuple[ndarray, ...]]:
        return [
            np.flipud(input[i, ...]),
            np.flipud(input[:, j, ...]),
            np.flipud(input[:, :, k, ...]),
        ]

    def plot(self) -> Figure:
        nrows, ncols = len(self.slices), 3  # one row for each slice position
        fig = plt.figure(figsize=(10, 10)) # fig = plt.figure(figsize=(13, 10))
        gs = gridspec.GridSpec(nrows, ncols)

        for i in range(0, nrows):
            ax1 = plt.subplot(gs[i * 3])
            ax2 = plt.subplot(gs[i * 3 + 1])
            ax3 = plt.subplot(gs[i * 3 + 2])
            axes = ax1, ax2, ax3
            self.plot_row(self.slices[i], axes)
            for axis in axes:
                if i == 0:
                    axis.set_title(self.title[0])
                elif i ==1:
                    axis.set_title(self.title[1])
                elif i ==2:
                    axis.set_title(self.title[2])
                elif i ==3:
                    axis.set_title(self.title[3])
                elif i ==4:
                    axis.set_title(self.title[4])
                elif i ==5:
                    axis.set_title(self.title[5])
            cm = plt.get_cmap('bone')
            if i ==0:
                # plt.subplots_adjust(bottom=0., right=0.92, top=1.)
                cax = plt.axes([0.93, 0.82, 0.010, 0.11]) #[left, bottom, width, height]
                sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=self.imax1, vmax=self.imin1))
                cbar=plt.colorbar(sm,cax)
            elif i==1:
                cax = plt.axes([0.93, 0.67, 0.010, 0.11])
                sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=self.imax2, vmax=self.imin2))
                cbar=plt.colorbar(sm,cax)
            elif i==2:
                cax = plt.axes([0.93, 0.51, 0.010, 0.11])
                sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=self.imax3, vmax=self.imin3))
                cbar=plt.colorbar(sm,cax)
            elif i==3:
                cax = plt.axes([0.93, 0.35, 0.010, 0.11])
                sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=self.imax4, vmax=self.imin4))
                cbar=plt.colorbar(sm,cax)
            elif i==4:
                cax = plt.axes([0.93, 0.20, 0.010, 0.11])
                sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=self.imax5, vmax=self.imin5))
                cbar=plt.colorbar(sm,cax)
            elif i==5:
                cax = plt.axes([0.93, 0.06, 0.010, 0.11])
                sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=self.imax6, vmax=self.imin6))
                cbar=plt.colorbar(sm,cax)
        plt.tight_layout(pad=3,h_pad=0.0, w_pad=0.0001) # plt.tight_layout(pad=3, h_pad=0.0, w_pad=0.1)
        # plt.imshow(fig)
        
        fig.suptitle(f'Parametric map images from EC metrics (Subject-{self.subject+1})', fontsize=16)
        return fig

    def plot_row(self, slices: List, axes: Tuple[Any, Any, Any]) -> None:
        for (slice_, axis) in zip(slices, axes):
            imgs = [img for img in slice_]
            # imgs = np.concatenate(imgs, axis=1)
            axis.imshow(imgs, cmap="bone", alpha=0.8, vmin=0, vmax=255, extent=(-0.5, 2-0.5, 1.50-0.5, -0.5)) # If raw then vmax=1; if scaled then vmax=255
            axis.grid(False)
            axis.invert_xaxis()
            axis.invert_yaxis()
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xticklabels([])
            axis.set_yticklabels([])


def generate_fig(
    ratio_ec_data: Union[ndarray, ndarray],
    ratio_diff_ec_data: Union[ndarray, ndarray],
    ratio_sign_ec_data: Union[ndarray, ndarray],
    ratio_diff_sign_ec_data: Union[ndarray, ndarray],
    inter_union_vox_ec_data: Union[ndarray, ndarray],
    inter_union_distance_ec_data: Union[ndarray, ndarray],
    subject: int,
) -> None:
    brainSlice = BrainSlices(
        ratio_ec_data,
        ratio_diff_ec_data,
        ratio_sign_ec_data,
        ratio_diff_sign_ec_data,
        inter_union_vox_ec_data,
        inter_union_distance_ec_data,
        subject,
    )

    fig = brainSlice.plot()

    filename = f"ec_method_image_sub_{subject+1}_raw.png"
    outfile = PARAMETRIC_MAP / filename
    fig.savefig(outfile, format='png')
    filename = f'ec_method_image_sub_{subject+1}_raw.pdf'
    outfile = PARAMETRIC_MAP / filename
    fig.savefig(outfile, dpi=120, format='pdf', bbox_inches='tight')
    
    fig.subplots_adjust(right=0.8, top=None, wspace=None, hspace=None)
    plt.close()


if __name__ == "__main__":

    for subject in range(0, 7):
        filename = f"all_method_ec_subject_{subject}.npz"
        filepath = TEST_PREDICTION / filename

        with np.load(filepath) as data:
            ratio_ec_data = data['ratio']
            ratio_diff_ec_data = data['ratio_diff']
            ratio_sign_ec_data = data['ratio_sign']
            ratio_diff_sign_ec_data = data['ratio_diff_sign']
            inter_union_vox_ec_data = data['inter_union_vox']
            inter_union_distance_ec_data = data['inter_union_distance'] # change to "inter_union_distance"
        generate_fig(ratio_ec_data, ratio_diff_ec_data, ratio_sign_ec_data, ratio_diff_sign_ec_data, 
                    inter_union_vox_ec_data, inter_union_distance_ec_data, subject=subject)
