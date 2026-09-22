import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_contour_snapshot(
    var,
    time,
    x,
    y,
    index=0,
    scale="global",
    levels=20,
    cmap="RdBu_r",
    label=None,
    ax=None,
):
    """
    Plot a contourf snapshot of a scalar field.

    Parameters
    ----------
    var : ndarray, shape (nt, nx, ny)
        Scalar field to plot.

    time : array_like, shape (nt,)
        Time coordinate.

    x : array_like, shape (nx,)
        x coordinate.

    y : array_like, shape (ny,)
        y coordinate.

    index : int, default 0
        Time index to plot.

    scale : {"snapshot", "global"} or int, default "global"
        Colour scale:

        "snapshot"
            Use min/max from var[index].

        "global"
            Use min/max over the entire var array.

        int
            Use min/max from var[scale].

    levels : int, default 20
        Number of filled contour levels.

    cmap : str, default "RdBu_r"
        Matplotlib colormap.

    label : str or None
        Colourbar label.

    ax : matplotlib Axes or None
        Optional existing axes.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """

    var = np.asarray(var)
    time = np.asarray(time)
    x = np.asarray(x)
    y = np.asarray(y)

    nt, nx, ny = var.shape

    if len(time) != nt:
        raise ValueError("len(time) must equal var.shape[0]")
    if len(x) != nx:
        raise ValueError("len(x) must equal var.shape[1]")
    if len(y) != ny:
        raise ValueError("len(y) must equal var.shape[2]")

    if not 0 <= index < nt and index != -1:
        raise ValueError(f"index must be between 0 and {nt - 1}")

    # Determine colour limits
    if scale == "global":
        vmin = np.nanmin(var)
        vmax = np.nanmax(var)

    elif scale == "snapshot":
        vmin = np.nanmin(var[index])
        vmax = np.nanmax(var[index])

    elif isinstance(scale, (int, np.integer)):
        if not 0 <= scale < nt:
            raise ValueError(
                f"scale index must be between 0 and {nt - 1}"
            )
        vmin = np.nanmin(var[scale])
        vmax = np.nanmax(var[scale])

    else:
        raise ValueError(
            "scale must be 'snapshot', 'global', or a time index"
        )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    contour_levels = np.linspace(vmin, vmax, levels)

    # var is stored as (x, y), while matplotlib expects (y, x)
    field = var[index].T

    contour = ax.contourf(
        x,
        y,
        field,
        levels=contour_levels,
        cmap=cmap,
        extend="both",
    )

    cbar = fig.colorbar(contour, ax=ax)

    if label is not None:
        cbar.set_label(label)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"t = {time[index]:g}")

    return fig, ax

def plot_height_velocity_snapshot(
    h,
    u,
    v,
    time,
    x,
    y,
    index=0,
    scale="global",
    levels=20,
    cmap="RdBu_r",
    quiver_skip=1,
    quiver_scale=None,
    ax=None,
):
    """
    Plot a snapshot of shallow-water height with velocity vectors overlaid.

    Parameters
    ----------
    h, u, v : ndarray, shape (nt, nx, ny)
        Height and velocity fields, all on the same grid.

    time : array_like, shape (nt,)
        Time coordinate.

    x : array_like, shape (nx,)
        x coordinate.

    y : array_like, shape (ny,)
        y coordinate.

    index : int, default 0
        Time index to plot.

    scale : {"snapshot", "global"} or int, default "global"
        Colour scale for h.

        "snapshot"
            Use the min/max of h at the plotted time index.

        "global"
            Use min/max over the full h array.

        int
            Use min/max from h[scale].

    levels : int, default 20
        Number of contour levels.

    cmap : str, default "RdBu_r"
        Colormap for h.

    quiver_skip : int, default 1
        Plot every nth velocity vector in each direction.

    quiver_scale : float or None
        Passed to matplotlib's quiver scale argument.

    ax : matplotlib Axes or None
        Optional existing axes to plot on.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """

    h = np.asarray(h)
    u = np.asarray(u)
    v = np.asarray(v)
    time = np.asarray(time)
    x = np.asarray(x)
    y = np.asarray(y)

    if h.shape != u.shape or h.shape != v.shape:
        raise ValueError("h, u, and v must have the same shape")

    nt, nx, ny = h.shape

    if len(time) != nt:
        raise ValueError("len(time) must equal h.shape[0]")
    if len(x) != nx:
        raise ValueError("len(x) must equal h.shape[1]")
    if len(y) != ny:
        raise ValueError("len(y) must equal h.shape[2]")

    if (not 0 <= index < nt) and index != -1:
        raise ValueError(f"index must be between 0 and {nt - 1}")

    # Determine colour limits
    if scale == "global":
        vmin = np.nanmin(h)
        vmax = np.nanmax(h)

    elif scale == "snapshot":
        vmin = np.nanmin(h[index])
        vmax = np.nanmax(h[index])

    elif isinstance(scale, (int, np.integer)):
        if not 0 <= scale < nt:
            raise ValueError(
                f"scale index must be between 0 and {nt - 1}"
            )
        vmin = np.nanmin(h[scale])
        vmax = np.nanmax(h[scale])

    else:
        raise ValueError(
            "scale must be 'snapshot', 'global', or a time index"
        )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    contour_levels = np.linspace(vmin, vmax, levels)

    # Fields are stored as (x, y), so transpose for plotting
    h_field = h[index].T
    u_field = u[index].T
    v_field = v[index].T

    contour = ax.contourf(
        x,
        y,
        h_field,
        levels=contour_levels,
        cmap=cmap,
        extend="both",
    )

    fig.colorbar(contour, ax=ax, label="h")

    X, Y = np.meshgrid(x, y, indexing="xy")
    sl = slice(None, None, quiver_skip)

    ax.quiver(
        X[sl, sl],
        Y[sl, sl],
        u_field[sl, sl],
        v_field[sl, sl],
        scale=quiver_scale,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"t = {time[index]:g}")

    return fig, ax

def animate_contour(
    data,
    time,
    x,
    y,
    scale="global",
    levels=20,
    cmap="RdBu_r",
    interval=100,
):
    """
    Animate a 2D field with shape (time, x, y).

    Parameters
    ----------
    data : ndarray, shape (nt, nx, ny)
        Field to animate.

    time : array_like, shape (nt,)
        Time coordinate.

    x : array_like, shape (nx,)
        x coordinate.

    y : array_like, shape (ny,)
        y coordinate.

    scale : {"dynamic", "global"} or int, default "global"
        Determines the colour scale:

        "dynamic"
            Recalculate vmin/vmax for every frame.

        "global"
            Use the minimum and maximum over the entire data array.

        int
            Use the minimum and maximum from data[scale].

    levels : int, default 20
        Number of contour levels.

    cmap : str, default "RdBu_r"
        Matplotlib colormap.

    interval : int, default 100
        Delay between frames in milliseconds.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        Animation object.
    """

    data = np.asarray(data)
    time = np.asarray(time)
    x = np.asarray(x)
    y = np.asarray(y)

    nt, nx, ny = data.shape

    if len(time) != nt:
        raise ValueError("len(time) must equal data.shape[0]")
    if len(x) != nx:
        raise ValueError("len(x) must equal data.shape[1]")
    if len(y) != ny:
        raise ValueError("len(y) must equal data.shape[2]")

    # Determine fixed colour limits if required
    if scale == "global":
        fixed_vmin = np.nanmin(data)
        fixed_vmax = np.nanmax(data)

    elif isinstance(scale, (int, np.integer)):
        if not 0 <= scale < nt:
            raise ValueError(
                f"scale index must be between 0 and {nt - 1}"
            )
        fixed_vmin = np.nanmin(data[scale])
        fixed_vmax = np.nanmax(data[scale])

    elif scale == "dynamic":
        fixed_vmin = fixed_vmax = None

    else:
        raise ValueError(
            "scale must be 'dynamic', 'global', or a time index"
        )

    fig, ax = plt.subplots()

    # data has ordering (time, x, y), whereas contourf expects
    # an array corresponding to (y, x), hence the transpose.
    field = data[0].T

    if scale == "dynamic":
        vmin = np.nanmin(field)
        vmax = np.nanmax(field)
    else:
        vmin = fixed_vmin
        vmax = fixed_vmax

    contour_levels = np.linspace(vmin, vmax, levels)

    contour = ax.contourf(
        x,
        y,
        field,
        levels=contour_levels,
        cmap=cmap,
        extend="both",
    )

    cbar = fig.colorbar(contour, ax=ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title(f"t = {time[0]:g}")

    def update(frame):
        """Redraw the animation for the integer time index ``frame``.
        
        Replaces contour artists, refreshes the time title and any dynamic
        color scale, and updates velocity arrows when present. Returns the
        updated artists for the enclosing animation callback."""
        nonlocal contour

        field = data[frame].T

        # Remove previous contours
        for collection in contour.collections:
            collection.remove()

        if scale == "dynamic":
            vmin = np.nanmin(field)
            vmax = np.nanmax(field)
            current_levels = np.linspace(vmin, vmax, levels)
        else:
            current_levels = contour_levels

        contour = ax.contourf(
            x,
            y,
            field,
            levels=current_levels,
            cmap=cmap,
            extend="both",
        )

        if scale == "dynamic":
            cbar.update_normal(contour)

        title.set_text(f"t = {time[frame]:g}")

        return contour.collections + [title]

    anim = FuncAnimation(
        fig,
        update,
        frames=nt,
        interval=interval,
        blit=False,
    )

    return anim


def animate_height_velocity(
    h,
    u,
    v,
    time,
    x,
    y,
    scale="global",
    levels=20,
    cmap="RdBu_r",
    interval=100,
    quiver_skip=1,
    quiver_scale=None,
):
    """
    Animate shallow-water height with velocity vectors overlaid.

    Parameters
    ----------
    h, u, v : ndarray, shape (nt, nx, ny)
        Height and velocity fields, all on the same grid.

    time : array_like, shape (nt,)
        Time coordinate.

    x : array_like, shape (nx,)
        x coordinate.

    y : array_like, shape (ny,)
        y coordinate.

    scale : {"dynamic", "global"} or int, default "global"
        Colour scale for h.

        "dynamic"
            Recalculate min/max for every frame.

        "global"
            Use min/max over all values of h.

        int
            Use min/max from h[scale].

    levels : int, default 20
        Number of contour levels.

    cmap : str, default "RdBu_r"
        Colormap for h.

    interval : int, default 100
        Delay between frames in milliseconds.

    quiver_skip : int, default 1
        Plot every nth velocity vector in each direction.

    quiver_scale : float or None
        Passed to matplotlib quiver's scale argument.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """

    h = np.asarray(h)
    u = np.asarray(u)
    v = np.asarray(v)
    time = np.asarray(time)
    x = np.asarray(x)
    y = np.asarray(y)

    if h.shape != u.shape or h.shape != v.shape:
        raise ValueError("h, u, and v must have the same shape")

    nt, nx, ny = h.shape

    if len(time) != nt:
        raise ValueError("len(time) must equal h.shape[0]")
    if len(x) != nx:
        raise ValueError("len(x) must equal h.shape[1]")
    if len(y) != ny:
        raise ValueError("len(y) must equal h.shape[2]")

    # Determine contour scaling
    if scale == "global":
        fixed_vmin = np.nanmin(h)
        fixed_vmax = np.nanmax(h)

    elif isinstance(scale, (int, np.integer)):
        if not 0 <= scale < nt:
            raise ValueError(
                f"scale index must be between 0 and {nt - 1}"
            )
        fixed_vmin = np.nanmin(h[scale])
        fixed_vmax = np.nanmax(h[scale])

    elif scale == "dynamic":
        fixed_vmin = fixed_vmax = None

    else:
        raise ValueError(
            "scale must be 'dynamic', 'global', or a time index"
        )

    fig, ax = plt.subplots()

    # Grid for quiver
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Initial height field
    h_field = h[0].T

    if scale == "dynamic":
        vmin = np.nanmin(h_field)
        vmax = np.nanmax(h_field)
    else:
        vmin = fixed_vmin
        vmax = fixed_vmax

    contour_levels = np.linspace(vmin, vmax, levels)

    contour = ax.contourf(
        x,
        y,
        h_field,
        levels=contour_levels,
        cmap=cmap,
        extend="both",
    )

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("h")

    # Subsample velocity vectors
    sl = slice(None, None, quiver_skip)

    quiver = ax.quiver(
        X[sl, sl],
        Y[sl, sl],
        u[0].T[sl, sl],
        v[0].T[sl, sl],
        scale=quiver_scale,
        # Keep arrows above contours recreated on each animation frame.
        zorder=2,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title(f"t = {time[0]:g}")

    def update(frame):
        """Redraw the animation for the integer time index ``frame``.
        
        Replaces contour artists, refreshes the time title and any dynamic
        color scale, and updates velocity arrows when present. Returns the
        updated artists for the enclosing animation callback."""
        nonlocal contour

        h_field = h[frame].T

        # Remove old contour collections
 #       for collection in contour.collections:
 #           collection.remove()

        if scale == "dynamic":
            vmin = np.nanmin(h_field)
            vmax = np.nanmax(h_field)
            current_levels = np.linspace(vmin, vmax, levels)
        else:
            current_levels = contour_levels

        contour = ax.contourf(
            x,
            y,
            h_field,
            levels=current_levels,
            cmap=cmap,
            extend="both",
        )

        if scale == "dynamic":
            cbar.update_normal(contour)

        quiver.set_UVC(
            u[frame].T[sl, sl],
            v[frame].T[sl, sl],
        )

        title.set_text(f"t = {time[frame]:g}")

#        return contour.collections + [quiver, title]

    anim = FuncAnimation(
        fig,
        update,
        frames=nt,
        interval=interval,
        blit=False,
    )

    return anim

