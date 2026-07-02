import numpy as np
import spherical as sf
from .. import Inertial
from .. import WaveformModes

class AsymptoticBondiData:
    """Class to store asymptotic Bondi data

    This class stores time data, along with the corresponding values of psi0 through psi4 and sigma.
    For simplicity, the data are stored as one contiguous array.  That is, *all*
    values are stored at all times, even if they are zero, and all Modes objects
    are stored with ell_min=0, even when their spins are not zero.

    The single contiguous array is then viewed as 6 separate ModesTimeSeries
    objects, which enables them to track their spin weights, and provides
    various convenient methods like `eth` and `ethbar`; `dot` and `ddot` for
    time-derivatives; `int` and `iint` for time-integrations; `norm` to take the
    norm of a function over the sphere; `bar` for conjugation of the functions
    (which is different from just conjugating the mode weights); etc.  It also
    handles algebra correctly -- particularly addition (which is disallowed when
    the spin weights differ) and multiplication (which can be delicate with
    regards to the resulting ell values).

    This may lead to some headaches when the user tries to do things that are
    disabled by Modes objects.  The goal is to create headaches if and only if
    the user is trying to do things that really should never be done (like
    conjugating mode weights, rather than the underlying function; adding modes
    with different spin weights; etc.).  Please open issues for any situations
    that don't meet this standard.

    This class also provides various convenience methods for computing things
    like the mass aspect, the Bondi four-momentum, the Bianchi identities, etc.

    """

    def __init__(self, strain=None, psi0=None, psi1=None, psi2=None, psi3=None, psi4=None, frameType=Inertial):
        """Create new storage for asymptotic Bondi data

        Parameters
        ==========
        strain: int or array_like
            Times at which the data will be stored.  If this is an int, an empty array of that size
            will be created.  Otherwise, this must be a 1-dimensional array of floats.
        ell_max: int
            Maximum ell value to be stored
        multiplication_truncator: callable [defaults to `sum`, even though `max` is nicer]
            Function to be used by default when multiplying Modes objects
            together.  See the documentation for spherical.Modes.multiply for
            more details. The default behavior with `sum` is the most correct
            one -- keeping all ell values that result -- but also the most
            wasteful, and very likely to be overkill.  The user should probably
            always use `max`.  (Unfortunately, this must remain an opt-in
            choice, to ensure that the user is aware of the situation.)

        """

        self.frame = np.array([])
        self.frameType = frameType
        self._psi0 = psi0
        self._psi1 = psi1
        self._psi2 = psi2
        self._psi3 = psi3
        self._psi4 = psi4
        self._strain = strain

        self.validate_fields()
        self.validate_times()

    @property
    def time(self):
        return self._time

    t = time

    @property
    def n_times(self):
        return self.time.size

    @property
    def strain(self):
        if self._strain is None:
            raise AttributeError("Strain data has not been provided.")
        else:
            return self._strain

    @strain.setter
    def strain(self, strain_prm):
        self._strain = strain_prm
        return self.strain

    h = strain

    @property
    def has_strain(self):
        return self.strain is not None

    @property
    def sigma(self):
        return 0.5 * self.h.bar

    @property
    def psi4(self):
        if self._psi4 is None:
            self._psi4 = -self.sigma.bar.ddot
        return self._psi4

    @psi4.setter
    def psi4(self, psi4prm):
        self._psi4 = psi4prm
        return self.psi4

    @property
    def has_psi4(self):
        return self.psi4 is not None

    @property
    def psi3(self):
        if self._psi3 is None:
            self._psi3 = -self.sigma.bar.dot.eth
        return self._psi3

    @psi3.setter
    def psi3(self, psi3prm):
        self._psi3 = psi3prm
        return self.psi3

    @property
    def has_psi3(self):
        return self.psi3 is not None

    @property
    def psi2(self):
        if self._psi2 is None:
            raise AttributeError("psi2 data has not been provided.")
        else:
            return self._psi2

    @psi2.setter
    def psi2(self, psi2prm):
        self._psi2 = psi2prm
        return self.psi2

    @property
    def has_psi2(self):
        return self.psi2 is not None

    @property
    def psi1(self):
        if self._psi1 is None:
            raise AttributeError("psi1 data has not been provided.")
        else:
            return self._psi1

    @psi1.setter
    def psi1(self, psi1prm):
        self._psi1 = psi1prm
        return self.psi1

    @property
    def has_psi1(self):
        return self.psi1 is not None

    @property
    def psi0(self):
        return self._psi0

    @psi0.setter
    def psi0(self, psi0prm):
        self._psi0 = psi0prm
        return self.psi0

    @property
    def has_psi0(self):
        return self.psi0 is not None

    def validate_fields(self):
        """Check if the input fields are sensible."""
        fields = ["psi0", "psi1", "psi2", "psi3", "psi4"]
        fields_present = [getattr(self, f"has_{field}") for field in fields]

        for i, (name, present) in enumerate(zip(fields, fields_present)):
            if present:
                missing = [f for f, p in zip(fields[i + 1 :], fields_present[i + 1 :]) if not p]
                if missing:
                    raise ValueError(f"{name} is present but higher-order Weyl scalars are missing: {missing}")

    def validate_times(self):

        WM_ref = self._strain

        for field in ("psi0", "psi1", "psi2", "psi3", "psi4"):
            if getattr(self, f"has_{field}"):
                if not (WM_ref.t == getattr(self, field).t).all():
                    raise ValueError(
                        f"All fields i.e. Strain and Weyl scalar components must share the same set of times."
                        f"The data for {field} has a different set of times."
                    )

        self._time = WM_ref.t

    def copy(self):
        import copy

        new_abd = type(self)(
            strain=self.strain,
            psi0=self.psi0,
            psi1=self.psi1,
            psi2=self.psi2,
            psi3=self.psi3,
            psi4=self.psi4,
            frameType=self.frameType,
        )
        state = copy.deepcopy(self.__dict__)
        new_abd.__dict__.update(state)

        return new_abd

    def interpolate(self, new_times):
        new_abd = type(self)(
            strain=self.strain,
            psi0=self.psi0,
            psi1=self.psi1,
            psi2=self.psi2,
            psi3=self.psi3,
            psi4=self.psi4,
            frameType=self.frameType,
        )
        # interpolate waveform data
        for field in ("psi0", "psi1", "psi2", "psi3", "psi4", "strain"):
            if getattr(self, f"has_{field}"):
                setattr(new_abd, field, getattr(self, field).interpolate(new_times))

        # interpolate frame data if necessary
        if self.frame.shape[0] == self.n_times:
            import quaternion
            new_abd.frame = quaternion.squad(self.frame, self.t, new_times)

        return new_abd

    # Slicing
    def __getitem__(self, key):
        """
        Extract time slices of the asymptotic Bondi data efficiently.
        """
        # If key is a valid time slice or index, extract the corresponding
        # sliced data

        if not isinstance(key, (slice, int)):
            raise ValueError(f"Invalid key `{key}` of type `{type(key)}`.")

        new_abd = type(self)(
            strain=self.h,
            psi0=self.psi0,
            psi1=self.psi1,
            psi2=self.psi2,
            psi3=self.psi3,
            psi4=self.psi4,
            frameType=self.frameType,
        )

        for field in ("psi0", "psi1", "psi2", "psi3", "psi4", "strain"):
            if getattr(self, f"has_{field}"):
                setattr(new_abd, field, getattr(self, field)[key])

        if self.frame.shape[0] == self.n_times:
            new_abd.frame = self.frame[key]
        return new_abd

    from .constraints import (
        bondi_constraints,
        bondi_violations,
        bondi_violation_norms,
        bianchi_0,
        bianchi_1,
        bianchi_2,
        constraint_3,
        constraint_4,
        constraint_mass_aspect,
    )

    # from .from_initial_values import from_initial_values
    from .transformations import transform

    from .bms_charges import (
        mass_aspect,
        bondi_rest_mass,
        bondi_four_momentum,
        bondi_angular_momentum,
        CWWY_angular_momentum,
        bondi_dimensionless_spin,
        bondi_boost_charge,
        bondi_CoM_charge,
        supermomentum,
    )

    # from .map_to_superrest_frame import map_to_superrest_frame
    # from .map_to_abd_frame import map_to_abd_frame
