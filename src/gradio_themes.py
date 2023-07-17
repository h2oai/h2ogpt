from __future__ import annotations

from typing import Iterable

from gradio.themes.soft import Soft
from gradio.themes import Color, Size
from gradio.themes.utils import colors, sizes, fonts

h2o_yellow = Color(
    name="yellow",
    c50="#fffef2",
    c100="#fff9e6",
    c200="#ffecb3",
    c300="#ffe28c",
    c400="#ffd659",
    c500="#fec925",
    c600="#e6ac00",
    c700="#bf8f00",
    c800="#a67c00",
    c900="#664d00",
    c950="#403000",
)
h2o_gray = Color(
    name="gray",
    c50="#f8f8f8",
    c100="#e5e5e5",
    c200="#cccccc",
    c300="#b2b2b2",
    c400="#999999",
    c500="#7f7f7f",
    c600="#666666",
    c700="#4c4c4c",
    c800="#333333",
    c900="#191919",
    c950="#0d0d0d",
)


text_xsm = Size(
    name="text_xsm",
    xxs="4px",
    xs="5px",
    sm="6px",
    md="7px",
    lg="8px",
    xl="10px",
    xxl="12px",
)


spacing_xsm = Size(
    name="spacing_xsm",
    xxs="1px",
    xs="1px",
    sm="1px",
    md="2px",
    lg="3px",
    xl="5px",
    xxl="7px",
)


radius_xsm = Size(
    name="radius_xsm",
    xxs="1px",
    xs="1px",
    sm="1px",
    md="2px",
    lg="3px",
    xl="5px",
    xxl="7px",
)


class H2oTheme(Soft):
    def __init__(
            self,
            *,
            primary_hue: colors.Color | str = h2o_yellow,
            secondary_hue: colors.Color | str = h2o_yellow,
            neutral_hue: colors.Color | str = h2o_gray,
            spacing_size: sizes.Size | str = sizes.spacing_md,
            radius_size: sizes.Size | str = sizes.radius_md,
            text_size: sizes.Size | str = sizes.text_lg,
            font: fonts.Font
            | str
            | Iterable[fonts.Font | str] = (
                fonts.GoogleFont("Montserrat"),
                "ui-sans-serif",
                "system-ui",
                "sans-serif",
            ),
            font_mono: fonts.Font
            | str
            | Iterable[fonts.Font | str] = (
                fonts.GoogleFont("IBM Plex Mono"),
                "ui-monospace",
                "Consolas",
                "monospace",
            ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            link_text_color="#3344DD",
            link_text_color_hover="#3344DD",
            link_text_color_visited="#3344DD",
            link_text_color_dark="#74abff",
            link_text_color_hover_dark="#a3c8ff",
            link_text_color_active_dark="#a3c8ff",
            link_text_color_visited_dark="#74abff",
            button_primary_text_color="*neutral_950",
            button_primary_text_color_dark="*neutral_950",
            button_primary_background_fill="*primary_500",
            button_primary_background_fill_dark="*primary_500",
            block_label_background_fill="*primary_500",
            block_label_background_fill_dark="*primary_500",
            block_label_text_color="*neutral_950",
            block_label_text_color_dark="*neutral_950",
            block_title_text_color="*neutral_950",
            block_title_text_color_dark="*neutral_950",
            block_background_fill_dark="*neutral_950",
            body_background_fill="*neutral_50",
            body_background_fill_dark="*neutral_900",
            background_fill_primary_dark="*block_background_fill",
            block_radius="0 0 8px 8px",
            checkbox_label_text_color_selected_dark='#000000',
            #checkbox_label_text_size="*text_xs",  # too small for iPhone etc. but good if full large screen zoomed to fit
            checkbox_label_text_size="*text_sm",
            #radio_circle="""url("data:image/svg+xml,%3csvg viewBox='0 0 32 32' fill='white' xmlns='http://www.w3.org/2000/svg'%3e%3ccircle cx='32' cy='32' r='1'/%3e%3c/svg%3e")""",
            #checkbox_border_width=1,
            #heckbox_border_width_dark=1,
        )


class SoftTheme(Soft):
    def __init__(
            self,
            *,
            primary_hue: colors.Color | str = colors.indigo,
            secondary_hue: colors.Color | str = colors.indigo,
            neutral_hue: colors.Color | str = colors.gray,
            spacing_size: sizes.Size | str = sizes.spacing_md,
            radius_size: sizes.Size | str = sizes.radius_md,
            text_size: sizes.Size | str = sizes.text_md,
            font: fonts.Font
            | str
            | Iterable[fonts.Font | str] = (
                fonts.GoogleFont("Montserrat"),
                "ui-sans-serif",
                "system-ui",
                "sans-serif",
            ),
            font_mono: fonts.Font
            | str
            | Iterable[fonts.Font | str] = (
                fonts.GoogleFont("IBM Plex Mono"),
                "ui-monospace",
                "Consolas",
                "monospace",
            ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            checkbox_label_text_size="*text_sm",
        )


h2o_logo = '<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%"' \
           ' viewBox="0 0 600.28 600.28"><defs><style>.cls-1{fill:#fec925;}.cls-2{fill:#161616;}.cls-3{fill:' \
           '#54585a;}</style></defs><g id="Fill-1"><rect class="cls-1" width="600.28" height="600.28" ' \
           'rx="23.24"/></g><path class="cls-2" d="M174.33,246.06v92.78H152.86v-38H110.71v38H89.24V246.06h21.' \
           '47v36.58h42.15V246.06Z"/><path class="cls-2" d="M259.81,321.34v17.5H189.7V324.92l35.78-33.8c8.22-7.' \
           '82,9.68-12.59,9.68-17.09,0-7.29-5-11.53-14.85-11.53-7.95,0-14.71,3-19.21,9.27L185.46,261.7c7.15-10' \
           '.47,20.14-17.23,36.84-17.23,20.68,0,34.46,10.6,34.46,27.44,0,9-2.52,17.22-15.51,29.29l-21.33,20.14Z"' \
           '/><path class="cls-2" d="M268.69,292.45c0-27.57,21.47-48,50.76-48s50.76,20.28,50.76,48-21.6,48-50.' \
           '76,48S268.69,320,268.69,292.45Zm79.78,0c0-17.63-12.46-29.69-29-29.69s-29,12.06-29,29.69,12.46,29.69' \
           ',29,29.69S348.47,310.08,348.47,292.45Z"/><path class="cls-3" d="M377.23,326.91c0-7.69,5.7-12.73,12.' \
           '85-12.73s12.86,5,12.86,12.73a12.86,12.86,0,1,1-25.71,0Z"/><path class="cls-3" d="M481.4,298.15v40.' \
           '69H462.05V330c-3.84,6.49-11.27,9.94-21.74,9.94-16.7,0-26.64-9.28-26.64-21.61,0-12.59,8.88-21.34,30.' \
           '62-21.34h16.43c0-8.87-5.3-14-16.43-14-7.55,0-15.37,2.51-20.54,6.62l-7.43-14.44c7.82-5.57,19.35-8.' \
           '62,30.75-8.62C468.81,266.47,481.4,276.54,481.4,298.15Zm-20.68,18.16V309H446.54c-9.67,0-12.72,3.57-' \
           '12.72,8.35,0,5.16,4.37,8.61,11.66,8.61C452.37,326,458.34,322.8,460.72,316.31Z"/><path class="cls-3"' \
           ' d="M497.56,246.06c0-6.49,5.17-11.53,12.86-11.53s12.86,4.77,12.86,11.13c0,6.89-5.17,11.93-12.86,' \
           '11.93S497.56,252.55,497.56,246.06Zm2.52,21.47h20.68v71.31H500.08Z"/></svg>'


def get_h2o_title(title, description):
    # NOTE: Check full width desktop, smallest width browser desktop, iPhone browsers to ensure no overlap etc.
    return f"""<div style="float:left; justify-content:left; height: 80px; width: 195px; margin-top:0px">
                    {description}
                </div>
                <div style="display:flex; justify-content:center; margin-bottom:30px; margin-right:330px;">
                    <div style="height: 60px; width: 60px; margin-right:20px;">{h2o_logo}</div>
                    <h1 style="line-height:60px">{title}</h1>
                </div>
                <div style="float:right; height: 80px; width: 80px; margin-top:-100px">
                    <img src="https://raw.githubusercontent.com/h2oai/h2ogpt/main/docs/h2o-qr.png">
                </div>
                """


def get_simple_title(title, description):
    return f"""{description}<h1 align="center"> {title}</h1>"""


def get_dark_js():
    return """() => {
        if (document.querySelectorAll('.dark').length) {
            document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
        } else {
            document.querySelector('body').classList.add('dark');
        }
    }"""
