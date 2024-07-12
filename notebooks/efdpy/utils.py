import numpy as np
from glob import glob
from datetime import datetime, timedelta

CSES_DATA_TABLE = {
    "EFD": {"1": "ULF", "2": "ELF", "3": "VLF", "4": "HF"},
    "HPM": {"1": "FGM1", "2": "FGM2", "3": "CDSM", "5": "FGM1Hz"},
    "SCM": {"1": "ULF", "2": "ELF", "3": "VLF"},
    "LAP": {"1": "50mm", "2": "10mm"},
    "PAP": {"0": ""},
    "HEP": {"1": "P_L", "2": "P_H", "3": "D", "4": "P_X"},
}

CSES_SAMPLINGFREQS = {
    "EFD_ULF": 125.0,
    "EFD_ELF": 5000.0,
    "EFD_VLF": 50000.0,
    "SCM_ULF": 1024.0,
    "SCM_ELF": 10240.0,
    "SCM_VLF": 51200.0,
    "LAP_50mm": 1 / 3,
    "PAP_": 1.0,
    "HPM_FGM1Hz": 1.0,
    "HEP": 1.0,
}

CSES_PACKETSIZE = {
    "EFD_ULF": 256,
    "EFD_ELF": 2048,
    "EFD_VLF": 2048,
    "EFD_HF": 2048,
    "SCM_ULF": 4096,
    "SCM_ELF": 4096,
    "SCM_VLF": 4096,
    "LAP_50mm": 1,
    "PAP_": 1,
    "HPM_FGM1Hz": 1,
    "HEP": 1,
}


def parse_CSES_filename(filename):

    fl_list = filename.split("_")
    out = {}
    if len(filename) == 66:
        out["Satellite"] = fl_list[0] + fl_list[1]
        out["Instrument"] = fl_list[2]
        try:
            out["Data Product"] = CSES_DATA_TABLE[fl_list[2]][fl_list[3]]
        except:
            out["Data Product"] = "Unknown"
        out["Instrument No."] = fl_list[3]
        out["Data Level"] = fl_list[4]
        out["orbitn"] = fl_list[6]
        out["year"] = fl_list[7][0:4]
        out["month"] = fl_list[7][4:6]
        out["day"] = fl_list[7][6:8]
        out["time"] = fl_list[8][0:2] + ":" + fl_list[8][2:4] + ":" + fl_list[8][4:6]
        out["t_start"] = datetime(
            int(out["year"]),
            int(out["month"]),
            int(out["day"]),
            int(fl_list[8][0:2]),
            int(fl_list[8][2:4]),
            int(fl_list[8][4:6]),
        )
        out["t_end"] = datetime(
            int(fl_list[9][0:4]),
            int(fl_list[9][4:6]),
            int(fl_list[9][6:8]),
            int(fl_list[10][0:2]),
            int(fl_list[10][2:4]),
            int(fl_list[10][4:6]),
        )
    elif len(filename) == 69:
        out["Satellite"] = fl_list[0] + "_01"
        out["Instrument"] = fl_list[1]
        out["Data Product"] = fl_list[2]
        out["Data Level"] = fl_list[-2]
        out["orbitn"] = fl_list[3]
        out["year"] = fl_list[4][0:4]
        out["month"] = fl_list[4][4:6]
        out["day"] = fl_list[4][6:8]
        out["time"] = fl_list[5][0:2] + ":" + fl_list[5][2:4] + ":" + fl_list[5][4:6]
        out["t_start"] = datetime(
            int(out["year"]),
            int(out["month"]),
            int(out["day"]),
            int(fl_list[5][0:2]),
            int(fl_list[5][2:4]),
            int(fl_list[5][4:6]),
        )
        out["t_end"] = datetime(
            int(fl_list[6][0:4]),
            int(fl_list[6][4:6]),
            int(fl_list[6][6:8]),
            int(fl_list[7][0:2]),
            int(fl_list[7][2:4]),
            int(fl_list[7][4:6]),
        )

    return out


def versetime_to_utc(versetime, t0=(2009, 1, 1)):
    """
    convert versetime to utc time
    output is a datetime object
    """

    vt0 = datetime(t0[0], t0[1], t0[2])

    return datetime(t0[0], t0[1], t0[2]) + timedelta(milliseconds=versetime)


def add_packets(xbig, jumps, npacks, dt, fill_missing="sampling"):
    """
    fills gap according to fill_missing option
    xbig : 2D array of size nrows,packet_size
    jumps: 1D array of indices where you have jumps (of size NJUMPS)
    npacks: 1D integer array of size NJUMPS containing the number of missing packets
    dt : sampling time (if fill_missing == 'sampling')
    """
    nrows, packet_size = xbig.shape
    xout = xbig.copy()

    mask = np.ones(nrows, dtype=bool)
    # filling missing columns starting from the end
    for i, nx in zip(np.flipud(jumps), np.flipud(npacks)):

        # inserting packets
        xout = np.insert(xout, [i + 1] * nx, np.zeros(packet_size), axis=0)
        mask = np.insert(mask, [i + 1] * nx, False)
        # xout = np.insert(xout, jumps+1,np.zeros(packet_size),axis=0)
        if fill_missing == "sampling":
            xout[i + 1 : i + nx + 1] = (
                np.arange(nx * packet_size).reshape((nx, packet_size)) + packet_size
            ) * dt + xbig[i, 0]

    return xout, mask


def fill_missing_packets(data, time1, dt, fill_value=np.nan):
    """
    Some description of what this function does
    """
    npackets, packetsize = data.shape
    time = time1.flatten()
    jumps = np.where(np.diff(time) > dt * packetsize * 1.1)[0]

    # calculating gap in terms of number of packets missing
    dtjumps = np.diff(time)[jumps]
    npacks = np.rint(dtjumps / (packet_size * dt)).astype(int) - 1
    # filling with linear interpolation
    t_new, mask_old = add_packets(time, jumps, npacks, dt)
    outdata = np.zeros(t_new.shape, dtype=data.dtype)
    outdata[mask_old] = data
    outdata[~mask_old] = fill_value
    return outdata
