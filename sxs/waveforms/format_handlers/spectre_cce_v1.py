"""Functions to load and save waveforms in SpECTRE CCE v1 format"""

from .. import WaveformModes
from . import rotating_paired_diff_multishuffle_bzip2 as rpdmb
import numpy as np
from ...utilities.monotonicity import index_is_monotonic

def save(*args, **kwargs):
    raise NotImplementedError("Saving waveforms in SpECTRE CCE format (v1) is not supported")


def load(file_name, **kwargs):
    """Load a waveform in SpECTRE CCE format (version 1)

    Parameters
    ----------
    file_name : str or Path
        Relative or absolute path to the input HDF5 file.  If this
        string contains but does not *end* with `'.h5'`, the remainder
        of the string is taken to be the group within the HDF5 file in
        which the data is stored.  Also note that a JSON file is
        expected in the same location, with `.h5` replaced by `.json`
        (and the corresponding data must be stored under the `group`
        key if relevant).

    Required keyword argument
    -------------------------
    group : str
        The group within the HDF5 file in which the data is stored.

    """
    import re
    from pathlib import Path
    import numpy as np
    import h5py
    import spherical

    # Make sure the file exists
    path = Path(file_name).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")

    # Get the group name
    group = kwargs.pop("group", None)
    if group is None:
        raise ValueError("The 'group' keyword argument is required")

    # Determine the data type and spin weight from the group name
    if "news" in group.lower():
        data_type = "news"
        spin_weight = -2
    elif "psi0" in group.lower():
        data_type = "psi0"
        spin_weight = 2
    elif "psi1" in group.lower():
        data_type = "psi1"
        spin_weight = 1
    elif "psi2" in group.lower():
        data_type = "psi2"
        spin_weight = 0
    elif "psi3" in group.lower():
        data_type = "psi3"
        spin_weight = -1
    elif "psi4" in group.lower():
        data_type = "psi4"
        spin_weight = -2
    elif "strain" in group.lower():
        data_type = "h"
        spin_weight = -2
    else:
        raise ValueError(f"Unrecognized data type in group name '{group}'")

    m_is_scaled_out = True
    r_is_scaled_out = True

    with h5py.File(str(path), "r") as h5:
        if group is not None:
            h5 = h5[group]
        legend = list(h5.attrs["Legend"])
        if len(legend) != h5.shape[1]:
            raise ValueError(
                f"Number of columns in the data ({h5.shape[1]}) "
                f"does not match the number of entries in the legend ({len(legend)})"
            )
        regex = re.compile(r"(Real|Imag) Y_([0-9]+),([-0-9]+)")

        time_index = legend.index("time")
        time = h5[:, time_index]
        indices = np.argsort(time)
        time = time[indices]
        lm = np.array(
            [[m[2], m[3]] for l in legend if (m:=regex.match(l))],
            dtype=int
        )
        ell_min = min(lm[:, 0])
        ell_max = max(lm[:, 0])

        data = np.zeros((len(time), spherical.Ysize(ell_min, ell_max)), dtype=complex)
        for i, legend_entry in enumerate(legend):
            if (match:=regex.match(legend_entry)):
                ell, m = int(match[2]), int(match[3])
                if match[1] == "Real":
                    data[:, spherical.Yindex(ell, m, ell_min=ell_min)] += h5[:, i][indices]
                elif match[1] == "Imag":
                    data[:, spherical.Yindex(ell, m, ell_min=ell_min)] += 1j * h5[:, i][indices]
                else:
                    raise ValueError(f"Unrecognized legend entry '{legend_entry}'")

        return WaveformModes(
            data,
            time=time,
            time_axis=0,
            modes_axis=1,
            frame_type="inertial",
            data_type=data_type,
            m_is_scaled_out=m_is_scaled_out,
            r_is_scaled_out=r_is_scaled_out,
            ell_min=ell_min,
            ell_max=ell_max,
            spin_weight=spin_weight,
        )

def create_abd_from_h5(
    file_format,
    convention="SpEC",
    radius=None,
    ch_mass=None,
    t_interpolate=None,
    t_0_superrest=None,
    padding_time=None,
    **kwargs,
):
    """Returns an AsymptoticBondiData object with waveform data loaded from specified H5 files.

    The AsymptoticBondiData class internally uses the Moreschi-Boyle conventions, see the following reference:
      O. Moreschi, On angular momentum at future null infinity, DOI:10.1088/0264-9381/3/4/006
    If necessary, the waveform data will be converted to the Moreschi-Boyle conventions when loaded.

    Parameters
    ----------
    file_format : 'SXS', 'SpECTRECCE_v1', 'RPDMB', or 'RPXMB'
        The H5 files may be in the one of the following file formats:
          * 'RPDMB' - Dimensionless waveforms compressed using the rotating_paired_diff_multishuffle_bzip2 format.
    convention : 'SpEC' or 'Moreschi-Boyle'
        The data conventions of the waveform data that will be loaded. This defaults to 'SpEC' since this will be
        most often used with 'SpEC' convention waveforms. The output convention is Moreschi-Boyle.
    radius : str, optional
        Worldtube radius used when running CCE; only needed for versions of SpECTRE before PR #5985.
        The time array of the worldtube is translated by this radius.
    ch_mass : float, optional
        Total Christodoulou mass of the system.
    t_interpolate : float array, optional
        Time array to interpolate to, e.g., the time array of the worldtube.
    t_0_superrest : float, optional
        When to map to the BMS superrest frame.
        Typically a few hundred M after the junk radiation is sufficient.
    padding_time : float, optional
        Time window length around t_0_superrest to use when mapping to the superrest frame.
        Typically a few hundred M or a few orbits is sufficient.

    Keyword Parameters
    ------------------
    Psi4 : str, optional
    Psi3 : str, optional
    Psi2 : str, optional
    Psi1 : str, optional
    Psi0 : str, optional
    h    : str, optional
        Path to H5 file containing the data. At least ONE the above waveform quantities is required.


    Returns
    -------
    AsymptoticBondiData

    """
    from ... import AsymptoticBondiData

    # Use case insensitive parameters
    file_format = file_format.lower()
    convention = convention.lower()

    # Load waveform data from H5 files into WaveformModes objects
    WMs = {}
    filenames = {}
    for data_label in ["Psi4", "Psi3", "Psi2", "Psi1", "Psi0", "h"]:
        if data_label in kwargs:
            filenames[data_label] = kwargs.pop(data_label)
            if file_format == "rpdmb":
                WMs[data_label] = rpdmb.load(filenames[data_label])
            else:
                raise ValueError(
                    f"File format '{file_format}' not recognized. "
                    "Must be 'RPDMB'."
                )

    if kwargs:
        import pprint
        warnings.warn("\nUnused kwargs passed to this function:\n{}".format(pprint.pformat(kwargs, width=1)))

    # Sanity check
    if not WMs:
        raise ValueError("No filenames have been provided. The data of at least one waveform quantity is required.")

    WM_ref = WMs[list(WMs.keys())[0]]
    for i in WMs:
        if not (WM_ref.t == WMs[i].t).all():
            raise ValueError(
                f"All waveforms must share the same set of times. The data "
                f"for {list(WMs.keys())[i].data_type_string} has a different set of times."
            )

    for i in WMs:
        # Make waveforms dimensionless (if they already are, does nothing)
        if ch_mass is not None:
            make_variable_dimensionless(WMs[i], ch_mass)

        # indices = index_is_monotonic(WMs[i].t)
        # WMs[i].t = WMs[i].t[indices]
        # WMs[i].data = WMs[i].data[indices]

    # Create an instance of AsymptoticBondiData
    abd = AsymptoticBondiData(strain_modes=WMs["h"], psi0=WMs["Psi0"], psi1=WMs["Psi1"], psi2=WMs["Psi2"], psi3=WMs["Psi3"], psi4=WMs["Psi4"])

    # Define factors to convert between input waveform convention and Moreschi-Boyle convention
    conversion_factor = {
        # "input convention" : [Ψ₀, Ψ₁, Ψ₂, Ψ₃, Ψ₄, h]
        "moreschi-boyle": [1, 1, 1, 1, 1, 1],
        "spec": [2, -np.sqrt(2), 1, -1 / np.sqrt(2), 0.5, 0.5],
    }

    # Load the WaveformModes data into the ABD object and convert to the
    # Moreschi-Boyle convention.
    # Check conventions.
    if "Psi4" in WMs:
        abd.psi4 = conversion_factor[convention][4] * WMs["Psi4"]
    if "Psi3" in WMs:
        abd.psi3 = conversion_factor[convention][3] * WMs["Psi3"]
    if "Psi2" in WMs:
        abd.psi2 = conversion_factor[convention][2] * WMs["Psi2"]
    if "Psi1" in WMs:
        abd.psi1 = conversion_factor[convention][1] * WMs["Psi1"]
    if "Psi0" in WMs:
        abd.psi0 = conversion_factor[convention][0] * WMs["Psi0"]
    # ABD uses the Newman-Penrose scalar sigma instead of the strain h, so we
    # have to take the complex conjugate.
    if "h" in WMs:
        abd.h = WMs["h"]

    # Interpolate to finer time array, if specified
    if t_interpolate is not None:
        idx1 = np.argmin(abs(t_interpolate - abd.t[0])) + 1
        idx2 = np.argmin(abs(t_interpolate - abd.t[-1])) + 1 - 1
        abd = abd.interpolate(t_interpolate[idx1:idx2])

    # # Map to superrest frame at some time over some window, if specified
    # if t_0_superrest is not None and padding_time is not None:
    #     abd, BMS, _ = abd.map_to_superrest_frame(t_0=t_0_superrest, padding_time=padding_time)

    return abd
