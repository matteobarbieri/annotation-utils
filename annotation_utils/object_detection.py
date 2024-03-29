def draw_corners(x, y, w, h, segment_length,
                 pil_draw, color=(0, 255, 0), width=3):
    """
    Draw four corners around at the edges of a bounding box

    Parameters
    ----------

    x : int
        The x coordinate of the top left corner of the bounding box.

    y : int
        The y coordinate of the top left corner of the bounding box.

    w : int
        The width of the bounding box.

    h : int
        The height of the bounding box.

    segment_length : int
        The length of the segments composing the corner.

    pil_draw : PIL.ImageDraw
        The ImageDraw object required for drawing on a pil image.

    color : tuple
        A 3-tuple representing the RGB values of the lines' color. The default
        is (0, 255, 0) (green).

    width : int
        The width of the segments composing the corner.

    >>> from PIL import Image, ImageDraw
    >>> pil_image = Image.open('/path/to/image.png')
    >>> pil_draw = ImageDraw.Draw(pil_image)
    >>> draw_corners(30, 40, 100, 60, 12, pil_draw)
    """

    # Draw two lines for each corner

    # NW
    pil_draw.line([x, y, x+segment_length, y], fill=color, width=width)
    pil_draw.line([x, y, x, y+segment_length], fill=color, width=width)

    # NE
    pil_draw.line([x+w-segment_length, y, x+w, y], fill=color, width=width)
    pil_draw.line([x+w, y, x+w, y+segment_length], fill=color, width=width)

    # SE
    pil_draw.line([x+w-segment_length, y+h, x+w, y+h], fill=color, width=width)
    pil_draw.line([x+w, y+h-segment_length, x+w, y+h], fill=color, width=width)

    # SW
    pil_draw.line([x, y+h, x+segment_length, y+h], fill=color, width=width)
    pil_draw.line([x, y+h-segment_length, x, y+h], fill=color, width=width)


def annotate_object(x, y, w, h, label_text, pil_draw,
                    font, padding, segment_length_ratio, line_length_ratio,
                    line_width,
                    outline_color, text_color, text_bg_color=(255, 255, 255),
                    scale=1,
                    object_symbol=None, font_symbol=None,
                    bg_color_symbol=None):
    """
    Draws corners around the identified object, plus a label and optionally a
    symbol.

    Parameters
    ----------

    x : int
        The x coordinate of the top left corner of the bounding box.

    y : int
        The y coordinate of the top left corner of the bounding box.

    w : int
        The width of the bounding box.

    h : int
        The height of the bounding box.

    label_text : str
        The label displayed close to the bounding box.

    pil_draw : PIL.ImageDraw
        The ImageDraw object required for drawing on a pil image.

    font : PIL.ImageFont
        The font used to write the label text.

    padding : int
        The padding of the text w.r.t. the boxes where is written inside.

    segment_length_ratio,
        Between 0 and 1, the length of the segments composing the corners.

    line_length_ratio : float
        Between 0 and 1, the length of the central top line

    line_width : int
        The width of all lines used in the annotation.

    outline_color : tuple
        A 3-tuple representing the RGB values of the outline color.

    text_color : tuple
        A 3-tuple representing the RGB values of the text. The default is
        (0, 0, 0) (black).

    text_bg_color : tuple
        A 3-tuple representing the RGB values of the background color of the
        box where the label is written. The default is (255, 255, 255) (white).

    scale : int
        The scale of annotations, used to change the size of boxes and symbols.
        The default value is 1, which should be ok for full HD or 2k
        images. For 4k images use 2.

    object_symbol : str or None
        A character representing an icon. Can be anything really but should be
        chosen by one of those fonts containing special symbols, such as one
        from the NerdFonts family.
        See https://www.nerdfonts.com for several examples.
        If `None`, the symbol (and its square) are not drawn on the image.
        The default value is `None`.

    font_symbol : PIL.ImageFont or None
        The font used to draw the special symbols.
        Only required if parameter object_symbol is not None.
        The default value is `None`.

    bg_color_symbol : tuple or None
        A 3-tuple representing the RGB values of the background of the box
        where the symbols is drawn.
        Only required if parameter object_symbol is not None.
        The default value is `None`.

    >>> # Import all required modules
    >>> from PIL import Image, ImageDraw, ImageFont

    >>> # Load the original image from disk
    >>> pil_image = Image.open('/path/to/image.png')

    >>> # Instantiate the ImageDraw object to draw on the image
    >>> pil_draw = ImageDraw.Draw(pil_image)

    >>> # Load the font used to write object labels
    >>> font = ImageFont.truetype('/path/tp/font.ttf', size=30)

    >>> # Load the font used for symbols (e.g. any of the Nerdfont) (optional)
    >>> font_symbol = ImageFont.truetype('/path/tp/font_symbol.ttf', size=40)

    >>> annotate_object(
            x, y, w, h  # Bounding box coordinates
            label_text, pil_draw,
            font,
            10,  # padding
            0.15,  # segment_ratio
            0.4,  # line_length_ratio
            3, # line_width
            (0, 255, 0),  # outline color, (green in this case)
            (0, 0, 0),  # text color (black)
            (255, 255, 255),  # label box background color (white)
            2, #  scale parameters, 1 for full HD images, 2 for 4k,
            object_symbol=" ",  # The actual symbol, as a string (requires special fonts)  # noqa
            font_symbol=font_large,
            bg_color_symbol=(0, 255, 0))
    """

    # Create a transparent version of the color
    # bg_color_transparent = (*outline_color, 128)

    # Adjust geometry to take padding into account
    x_p = x - padding
    y_p = y - padding
    w_p = w + 2 * padding
    h_p = h + 2 * padding

    # Compute length of corner segment in pixel
    segment_length = w_p * segment_length_ratio

    # Draw corners
    draw_corners(x_p, y_p, w_p, h_p, segment_length,
                 pil_draw, color=outline_color, width=line_width)

    # Compute the length of the horizontal line
    line_length = w_p * line_length_ratio
    line_height = 1.5 * line_length

    # Adapt line height to scale
    line_height = line_height if scale == 1 else line_height * 1.5

    # Compute x coordinate of the lp center
    x_c = x_p + w_p/2

    # Draw the horizontal line
    pil_draw.line([x_c - line_length/2, y_p, x_c + line_length/2, y_p],
                  fill=outline_color, width=line_width)

    # Draw the vertical line
    pil_draw.line([x_c, y_p-line_height, x_c, y_p],
                  fill=outline_color, width=line_width)

    # Compute coordinates for the label
    # x_symbol = x_c + 8
    x_symbol = x_c + 3 + scale * 5
    y_symbol = y_p - line_height

    y_text = y_symbol

    TEXT_BOX_HEIGHT = 30 * scale
    TEXT_BOX_WIDTH = (len(label_text)) * 21 * scale

    # Only draw symbol if passed as an argument, else skip
    if object_symbol is not None:

        # Draw the square for the symbol
        pil_draw.rectangle(
            [
                x_symbol,
                y_symbol,
                x_symbol+TEXT_BOX_HEIGHT,
                y_symbol+TEXT_BOX_HEIGHT],
            fill=outline_color,
            outline=(0, 0, 0))

        # Draw the symbol corresponding to the identified object
        pil_draw.text(
            (x_symbol + 3*scale, y_symbol - 3 - 5*scale),
            object_symbol,
            text_color, font=font_symbol)

        # Compute coordinates for the acutal LP text
        x_text = x_symbol + TEXT_BOX_HEIGHT + 5 * scale

    else:

        # Compute coordinates for the acutal LP text
        x_text = x_symbol

    # Draw the rectangle for the text
    pil_draw.rectangle(
        [x_text, y_text, x_text+TEXT_BOX_WIDTH, y_text + TEXT_BOX_HEIGHT],
        fill=text_bg_color,
        outline=(0, 0, 0))

    # Finally, write the actual LP text in the square
    pil_draw.text(
        # (x_text + 4, y_text - 2),  # for nerdfonts
        (x_text + 4*scale, y_text - 7*scale),  # for open sans
        label_text, text_color, font=font)
