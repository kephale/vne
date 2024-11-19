import napari
import torch
import umap

import numpy as np

import numpy.typing as npt
from typing import Dict, Optional, Tuple

from qtpy import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from vne.decoders.spatial import CartesianAxes


MAX_UMAP = 20.0


def process(
    model: torch.nn.Module,
    z: tuple,
    pose: float,
    *,
    device: torch.device = torch.device("cpu"),
    use_conv: bool = False,
) -> npt.NDArray:
    z = torch.tensor(np.array([z], dtype=np.float32), device=device)
    pose = torch.tensor(np.array([pose], dtype=np.float32), device=device)

    with torch.no_grad():
        x = model.decoder.forward(z, pose, use_final_convolution=use_conv)

    return x.squeeze().cpu().numpy()



def scale_from_slider(x, s):
    return 2 * ((x / 1000.0) - 0.5) * s


def scale_to_slider(x, s):
    return np.clip(((x + s) / (2 * s)) * 1000, 0, 1000).astype(int)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=6, height=6, dpi=100):
        with plt.style.context("dark_background"):
            fig = Figure(
                figsize=(width, height), dpi=dpi, frameon=False, facecolor="k"
            )
            self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class GenerativeAffinityVAEWidget(QtWidgets.QWidget):
    """A widget to allow interactivity with the AffinityVAE model.

    Parameters
    ----------
    napari_viewer : Viewer
    model : AffinityVAE
    latent_dims : int
    mapper : UMAP, optional
    reconstruction_layer_name : str
    device :

    """

    def __init__(
        self,
        napari_viewer: napari.Viewer,
        model: torch.nn.Module,
        *,
        latent_dims: int = 8,
        mapper: Optional[umap.UMAP] = None,
        reconstruction_layer_name: str = "Reconstruction",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self.viewer = napari_viewer
        self._model = model
        self._latent_dims = latent_dims
        self._mapper = mapper
        self._layer = self.viewer.layers[reconstruction_layer_name]
        self._device = device

        self._main_layout = QtWidgets.QVBoxLayout()
        self._tabs = QtWidgets.QTabWidget()
        self._widgets = {}

        self.add_pose_widget()
        self._main_layout.addWidget(self._tabs, stretch=0)
        self.add_latent_widget()
        self.add_manifold_widget()

        # Expand the main widget
        self.setLayout(self._main_layout)
        self._main_layout.addStretch(stretch=1)
        self.setMinimumWidth(400)

    def add_pose_widget(self) -> None:
        """Add widgets to manipulate the model pose space."""
        pose_axes = QtWidgets.QComboBox()
        pose_axes.addItems(["X", "Y", "Z"])

        pose_value = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        pose_value.setRange(0, 1000)
        pose_value.setValue(500)
        pose_value.setMinimumWidth(300)

        use_conv = QtWidgets.QCheckBox()
        use_conv.setText("Use final convolution")

        pose_widgets = {
            "axes": pose_axes,
            "theta": pose_value,
            "use_conv": use_conv,
        }

        self._widgets.update(pose_widgets)

        layout = QtWidgets.QFormLayout()
        for label, widget in pose_widgets.items():
            if label == "theta":
                widget.valueChanged.connect(self.update_reconstruction)
            if label != "use_conv":
                label_widget = QtWidgets.QLabel(label)
            else:
                label_widget = QtWidgets.QLabel()
            layout.addRow(label_widget, widget)

        pose_widget = QtWidgets.QGroupBox("Pose")
        pose_widget.setLayout(layout)

        self._main_layout.addWidget(pose_widget)

    def add_latent_widget(self) -> None:
        """Add widgets to manipulate the model latent space."""

        def _z_widget(idx):
            z_value = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            z_value.setRange(0, 1000)
            z_value.setValue(500)
            z_value.setMinimumWidth(300)
            return z_value

        latent_widgets = {
            f"z{idx}": _z_widget(idx) for idx in range(self._latent_dims)
        }

        self._widgets.update(latent_widgets)

        layout = QtWidgets.QFormLayout()
        for label, widget in latent_widgets.items():
            widget.valueChanged.connect(self.update_reconstruction)
            label_widget = QtWidgets.QLabel(label)
            layout.addRow(label_widget, widget)

        tab = QtWidgets.QWidget()
        tab.setLayout(layout)
        self._tabs.addTab(tab, "Latents")

    def add_manifold_widget(self) -> None:
        manifold_widget = MplCanvas()
        manifold_widget.axes.set_title("Latent manifold")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(manifold_widget)
        tab = QtWidgets.QWidget()
        tab.setLayout(layout)

        manifold_widget.figure.canvas.mpl_connect(
            "button_press_event", self.inverse_map_manifold_to_z
        )
        self._tabs.addTab(tab, "Manifold")

        self._widgets.update(
            {
                "manifold": manifold_widget,
            }
        )

    def get_pose(self) -> npt.NDArray:
        theta = scale_from_slider(self._widgets["theta"].value(), np.pi)
        axis = CartesianAxes[str(self._widgets["axes"].currentText())]
        return np.array([theta, *axis.value], dtype=np.float32)

    def get_state(self) -> Tuple[float, npt.NDArray]:
        pose = self.get_pose()
        z_values = np.array(
            [
                self._widgets[f"z{idx}"].value()
                for idx in range(self._latent_dims)
            ]
        )
        z = scale_from_slider(z_values, MAX_UMAP)
        return pose, z

    def inverse_map_manifold_to_z(self, event) -> None:
        test_pt = np.array([event.xdata, event.ydata]).reshape(1, -1)

        if self._mapper is not None:
            inv_transformed_points = self._mapper.inverse_transform(test_pt)
        else:
            inv_transformed_points = np.zeros((1, self._latent_dims))
            inv_transformed_points[:2] = test_pt

        transformed = [
            scale_to_slider(pt, MAX_UMAP)
            for pt in np.squeeze(inv_transformed_points)
        ]

        pose = self.get_pose()
        use_conv = self._widgets["use_conv"].isChecked()
        self._layer.data = process(
            self._model,
            inv_transformed_points,
            pose,
            device=self._device,
            use_conv=use_conv,
        )

        for idx in range(self._latent_dims):
            slider = self._widgets[f"z{idx}"]
            with QtCore.QSignalBlocker(slider):
                slider.setValue(transformed[idx])

    def update_reconstruction(self) -> None:
        pose, z = self.get_state()
        use_conv = self._widgets["use_conv"].isChecked()
        self._layer.data = process(
            self._model, z, pose, device=self._device, use_conv=use_conv
        )

        torch.mps.empty_cache()

    def set_embedding(
        self,
        embedding: npt.NDArray,
        *,
        labels: Optional[Dict[str, npt.NDArray]] = None,
    ) -> None:
        from scipy.stats import gaussian_kde
        from matplotlib import colormaps

        kernel = gaussian_kde(embedding.T)

        xmin, xmax = embedding[:, 0].min() - 1, embedding[:, 0].max() + 1
        ymin, ymax = embedding[:, 1].min() - 1, embedding[:, 1].max() + 1

        X, Y = np.meshgrid(
            np.linspace(xmin, xmax, 512),
            np.linspace(ymin, ymax, 512),
            indexing="xy",
        )
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)

        self._widgets["manifold"].axes.imshow(
            np.flipud(Z),
            extent=[xmin, xmax, ymin, ymax],
            cmap=colormaps.get_cmap("inferno"),
        )

        if labels is None:
            return

        with plt.style.context("dark_background"):
            for label, centroid in labels.items():
                self._widgets["manifold"].axes.text(
                    centroid[0], centroid[1], str(label[:4])
                )

        self._widgets["manifold"].axes.set_aspect("equal")
