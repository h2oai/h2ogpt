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
            background_fill_primary_dark="*block_background_fill",
            block_background_fill_dark="*neutral_950",
            block_border_width='1px',
            block_border_width_dark='1px',
            block_label_background_fill="*primary_300",
            block_label_background_fill_dark="*primary_600",
            block_label_text_color="*neutral_950",
            block_label_text_color_dark="*neutral_950",
            block_radius="0 0 8px 8px",
            block_title_text_color="*neutral_950",
            block_title_text_color_dark="*neutral_950",
            body_background_fill="*neutral_50",
            body_background_fill_dark="*neutral_900",
            border_color_primary="*neutral_100",
            border_color_primary_dark="*neutral_700",
            button_border_width="1px",
            button_border_width_dark="1px",
            button_primary_text_color="*neutral_950",
            button_primary_text_color_dark="*neutral_950",
            button_primary_background_fill="*primary_500",
            button_primary_background_fill_dark="*primary_500",
            button_secondary_background_fill_hover_dark="*primary_700",
            button_secondary_border_color="*primary_500",
            button_secondary_border_color_dark="*primary_500",
            button_secondary_border_color_hover_dark="*primary_700",
            checkbox_label_text_color_selected_dark='#000000',
            # checkbox_label_text_size="*text_xs",  # too small for iPhone etc. but good if full large screen zoomed to fit
            checkbox_label_text_size="*text_sm",
            # radio_circle="""url("data:image/svg+xml,%3csvg viewBox='0 0 32 32' fill='white' xmlns='http://www.w3.org/2000/svg'%3e%3ccircle cx='32' cy='32' r='1'/%3e%3c/svg%3e")""",
            # checkbox_border_width=1,
            # heckbox_border_width_dark=1,
            link_text_color="#3344DD",
            link_text_color_hover="#3344DD",
            link_text_color_visited="#3344DD",
            link_text_color_dark="#74abff",
            link_text_color_hover_dark="#a3c8ff",
            link_text_color_active_dark="#a3c8ff",
            link_text_color_visited_dark="#74abff",
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


def get_h2o_title(title, description, visible_h2ogpt_qrcode):
    # NOTE: Check full width desktop, smallest width browser desktop, iPhone browsers to ensure no overlap etc.
    ret = """<img src="https://h2o-release.s3.amazonaws.com/h2ogpt/happyHolidays-2023-banner.png" class="center">"""
    #ret = """<img src="https://h2o-release.s3.amazonaws.com/h2ogpt/banner-happyHolidaysFromH2O.jpg" class="center">"""
    if visible_h2ogpt_qrcode:
        ret += """
                <div style="float:right; height: 200px; width: 200px; margin-top:-200px">
                    <img src="https://raw.githubusercontent.com/h2oai/h2ogpt/main/docs/h2o-qr.png">
                </div>
                """
    return ret


def get_simple_title(title, description):
    return f"""{description}<h1 align="center"> {title}</h1>"""


def get_dark_js() -> str:
    return """
        if (document.querySelectorAll('.dark').length) {
            document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
        } else {
            document.querySelector('body').classList.add('dark');
        }
    """


def get_heap_js(heapAppId: str) -> str:
    return (
        """globalThis.window.heap=window.heap||[],heap.load=function(e,t){window.heap.appid=e,window.heap.config=t=t||{};var r=document.createElement("script");r.type="text/javascript",r.async=!0,r.src="https://cdn.heapanalytics.com/js/heap-"+e+".js";var a=document.getElementsByTagName("script")[0];a.parentNode.insertBefore(r,a);for(var n=function(e){return function(){heap.push([e].concat(Array.prototype.slice.call(arguments,0)))}},p=["addEventProperties","addUserProperties","clearEventProperties","identify","resetIdentity","removeEventProperty","setEventProperties","track","unsetEventProperty"],o=0;o<p.length;o++)heap[p[o]]=n(p[o])};"""
        f"""heap.load("{heapAppId}");""")


def wrap_js_to_lambda(num_params: int, *args: str) -> str:
    """
    Generates a JS code representing JS lambda that wraps all given '*args' code strings.
    The lambda function has number of parameters based on 'num_params' and returns them
    without modification in an array. Lambda with zero parameters returns an empty array.
    """
    params = ", ".join([f"p{i}" for i in range(num_params)])
    newline = "\n"
    return f"""
        ({params}) => {{
            {newline.join([a for a in args if a is not None])}
            return [{params}];
        }}
    """
