import numpy as np
from astropy import units as u
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits


def make_square(hdu, pad_value=0):

    hdu_square = hdu.copy()

    shape_diff = np.max(hdu_square.shape) - np.min(hdu_square.shape)
    square_img = np.ones((np.max(hdu_square.shape), np.max(hdu_square.shape))) * pad_value

    new_header = hdu_square.header.copy()

    if hdu_square.shape[0] > hdu_square.shape[1]:
        new_header['CRPIX1'] += int(shape_diff / 2)
        square_img[:, int(shape_diff / 2):int(shape_diff / 2 + hdu_square.shape[1])] = hdu_square.data
    else:
        new_header['CRPIX2'] += int(shape_diff / 2)
        square_img[int(shape_diff / 2):int(shape_diff / 2 + hdu_square.shape[0]), :] = hdu_square.data

    return fits.PrimaryHDU(square_img, new_header)


def cut_empty_rows(hdu, parallel_hdu=None, pad=0):

    hdu_trimmed = hdu.copy()
    parallel_hdu_trimmed = parallel_hdu.copy()

    # Find empty rows
    empty_x = np.all(hdu_trimmed.data == 0)

    # If the number of empty rows is larger than the desired padding, cut away rows and leave padding.
    if np.sum(empty_x) > pad:
        pad = int(np.round(pad))
        idx_false = [i for i, x in enumerate(empty_x) if not x]
        first_false = idx_false[0]
        last_false = idx_false[-1]
        empty_x[first_false:last_false] = False  # Make sure empty rows in the middle of the data are not affected
        last_false += 1
        if first_false - pad > 0:
            empty_x[first_false - pad:first_false] = False
        else:
            empty_x[0:first_false] = False
        empty_x[last_false:last_false + pad] = False

        hdu_trimmed.data = hdu_trimmed.data[~empty_x, :]
        if parallel_hdu:
            parallel_hdu_trimmed.data = parallel_hdu_trimmed.data[~empty_x, :]

        # Adjust the central pixel in the image header correspondingly
        pix_shift = [i for i, x in enumerate(empty_x) if not x][1] - 1
        if empty_x[0]:
            hdu_trimmed.header['CRPIX2'] -= pix_shift
        hdu_trimmed.header['NAXIS2'] = hdu.shape[0]
        if parallel_hdu:
            parallel_hdu_trimmed.header['CRPIX2'] -= pix_shift
            parallel_hdu_trimmed.header['NAXIS2'] = parallel_hdu_trimmed.shape[0]

    if parallel_hdu:
        return hdu_trimmed, parallel_hdu_trimmed
    else:
        return hdu_trimmed


def cut_empty_columns(hdu, parallel_hdu=None, pad=0):

    hdu_trimmed = hdu.copy()
    parallel_hdu_trimmed = parallel_hdu.copy()

    # Find empty columns
    empty_y = np.all(hdu_trimmed.data == 0)

    # If the number of empty columns is larger than the beamsize, cut away rows and leave as many as the beam size
    if np.sum(empty_y) > pad:
        pad = int(np.round(pad))
        idx_false = [i for i, x in enumerate(empty_y) if not x]
        first_false = idx_false[0]
        last_false = idx_false[-1]
        empty_y[first_false:last_false] = False  # Make sure empty rows in the middle of the data are not affected
        last_false += 1
        if first_false - pad > 0:
            empty_y[first_false - pad:first_false] = False
        else:
            empty_y[0:first_false] = False
        empty_y[last_false:last_false + pad] = False

        hdu_trimmed.data = hdu_trimmed.data[:, ~empty_y]
        if parallel_hdu:
            parallel_hdu_trimmed.data = parallel_hdu_trimmed.data[:, ~empty_y]

        # Adjust the header
        pix_shift = [i for i, x in enumerate(empty_y) if not x][1] - 1
        if empty_y[0]:
            hdu_trimmed.header['CRPIX1'] -= pix_shift
        hdu_trimmed.header['NAXIS1'] = hdu_trimmed.shape[1]
        if parallel_hdu:
            parallel_hdu_trimmed.header['CRPIX1'] -= pix_shift
            parallel_hdu_trimmed.header['NAXIS1'] = parallel_hdu_trimmed.shape[1]

    if parallel_hdu:
        return hdu_trimmed, parallel_hdu_trimmed
    else:
        return hdu_trimmed


def centre_data(hdu, centre, frame='sky', parallel_hdu=None, pad_value=0):
    
    hdu_centred = hdu.copy()
    parallel_hdu_centred = parallel_hdu.copy()

    # If frame == 'sky', find central pixel
    if frame == 'sky':
        w = wcs.WCS(hdu_centred.header, naxis=2)
        centre_sky = SkyCoord(centre[0], centre[1], unit=(u.deg, u.deg))
        centre_pix = wcs.utils.skycoord_to_pixel(centre_sky, w)
    elif frame == 'pixel':
        centre_pix = centre
    else:
        raise AttributeError('Please choose between "sky" and "pixel" for the keyword "frame".')

    # The amount the centre needs to shift to overlap with the central coordinates
    shift_x = int(np.round(centre_pix[1] - hdu_centred.shape[1] / 2))
    shift_y = int(np.round(centre_pix[0] - hdu_centred.shape[2] / 2))

    # Pad the image with twice the amount it has to shift, so that the new centre overlaps with the coordinates
    if shift_x > 0:
        temp = np.ones((hdu_centred.shape[0] + shift_x * 2, hdu_centred.shape[1])) * pad_value
        temp[0:hdu_centred.shape[0], :] = hdu_centred.data
        if parallel_hdu:
            temp_parallel = np.ones((parallel_hdu_centred.shape[0] + shift_x * 2, parallel_hdu_centred.shape[1])) * pad_value
            temp_parallel[0:parallel_hdu_centred.shape[0], :] = parallel_hdu_centred.data
    elif shift_x < 0:
        temp = np.ones((hdu_centred.shape[0] + abs(shift_x) * 2, hdu_centred.shape[1])) * pad_value
        temp[temp.shape[0] - hdu_centred.shape[0]:temp.shape[0], :] = hdu_centred.data
        if parallel_hdu:
            temp_parallel = np.ones((parallel_hdu_centred.shape[0] + abs(shift_x) * 2, parallel_hdu_centred.shape[1])) * pad_value
            temp_parallel[temp_parallel.shape[0] - parallel_hdu_centred.shape[0]:temp_parallel.shape[0], :] = parallel_hdu_centred.data
    else:
        temp = hdu_centred.data
        if parallel_hdu:
            temp_parallel = parallel_hdu_centred.data

    # Same in the y-direction
    if shift_y > 0:
        hdu_new = np.ones((temp.shape[0], temp.shape[1] + shift_y * 2)) * pad_value
        hdu_new[:, 0:temp.shape[1]] = temp
        if parallel_hdu:
            parallelhdu_new = np.ones((temp_parallel.shape[0], temp_parallel.shape[1] + shift_y * 2)) * pad_value
            parallelhdu_new[:, 0:temp_parallel.shape[1]] = temp_parallel
    elif shift_y < 0:
        hdu_new = np.ones((temp.shape[0], temp.shape[1] + abs(shift_y) * 2)) * pad_value
        hdu_new[:, hdu_new.shape[1] - temp.shape[1]:hdu_new.shape[1]] = temp
        if parallel_hdu:
            parallelhdu_new = np.ones((temp_parallel.shape[0], temp_parallel.shape[1] + abs(shift_y) * 2)) * pad_value
            parallelhdu_new[:, parallelhdu_new.shape[1] - temp_parallel.shape[1]:parallelhdu_new.shape[1]] = temp_parallel
    else:
        hdu_new = temp
        if parallel_hdu:
            parallelhdu_new = temp_parallel

    new_header = hdu_centred.header.copy()
    # Only change the CRPIX if the padding happens BEFORE the current CRPIX
    if shift_y < 0:
        new_header['CRPIX1'] = hdu.header['CRPIX1'] - 2 * shift_y
    if shift_x < 0:
        new_header['CRPIX2'] = hdu.header['CRPIX2'] - 2 * shift_x
    new_header['NAXIS1'] = hdu_new.shape[1]
    new_header['NAXIS2'] = hdu_new.shape[0]

    hdu_hdu = fits.PrimaryHDU(hdu_new, new_header)

    parallelhdu_hdu = fits.PrimaryHDU(parallelhdu_new, new_header)
    if parallel_hdu:
        parallelhdu_hdu.header['NAXIS3'] = parallel_hdu_centred.header['NAXIS3']

    if parallel_hdu:
        return hdu_hdu, parallelhdu_hdu
    else:
        return hdu_hdu


def trim_image(hdu, parallel_hdu=None, pad=0, centre=None, centre_frame='sky', square=False, pad_value=0):

    if parallel_hdu:
        hdu, parallel_hdu = cut_empty_rows(hdu, parallel_hdu=parallel_hdu, pad=pad)
        hdu, parallel_hdu = cut_empty_columns(hdu, parallel_hdu=parallel_hdu, pad=pad)
        if centre:
            hdu, parallel_hdu = centre_data(hdu=hdu, parallel_hdu=parallel_hdu, frame=centre_frame, pad_value=pad_value)
        if square:
            hdu = make_square(hdu=hdu, pad_value=pad_value)
            parallel_hdu = make_square(parallel_hdu, pad_value=pad_value)
        return hdu, parallel_hdu
    else:
        hdu = cut_empty_rows(hdu, parallel_hdu=parallel_hdu, pad=pad)
        hdu = cut_empty_columns(hdu, parallel_hdu=parallel_hdu, pad=pad)
        if centre:
            hdu = centre_data(hdu=hdu, parallel_hdu=parallel_hdu, frame=centre_frame, pad_value=pad_value)
        if square:
            hdu = make_square(hdu=hdu, pad_value=pad_value)
        return hdu

