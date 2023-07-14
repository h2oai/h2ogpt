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
            # checkbox_label_text_size="*text_xs",  # too small for iPhone etc. but good if full large screen zoomed to fit
            checkbox_label_text_size="*text_sm",
            # radio_circle="""url("data:image/svg+xml,%3csvg viewBox='0 0 32 32' fill='white' xmlns='http://www.w3.org/2000/svg'%3e%3ccircle cx='32' cy='32' r='1'/%3e%3c/svg%3e")""",
            # checkbox_border_width=1,
            # heckbox_border_width_dark=1,
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


h2o_logo = '<svg xmlns = "http://www.w3.org/2000/svg" width = "432" height = "288" viewBox = "0 0 432 288" xml: space = "preserve" >\
<path fill = "#00B388" d = "M48.492 73.322v32.943h114.691V73.322H48.492zm107.523 25.776H55.662V80.49h100.354v18.608z"/>\
<path fill = "#FFF" d = "M55.661 160.714h-7.168V127.77h7.168v13.019H68.47V127.77h7.168v32.944H68.47v-13.652H55.661v13.652zm36.264.526c-7.273 0-12.229-4.586-12.229-12.122 0-7.328 4.85-12.389 11.28-12.389 7.01 0 10.488 4.692 10.488 11.703v2.636h-14.97c.844 3.636 3.69 4.583 6.537 4.583 2.477 0 4.269-.525 6.482-1.896h.264v5.429c-1.896 1.371-4.585 2.056-7.852 2.056zm-5.43-14.969h8.539c-.157-2.424-1.159-4.111-3.953-4.111-2.108-.001-3.953.895-4.586 4.111zm33.681.632-3.689 13.811h-6.272l-7.169-23.193v-.263h6.852l3.901 13.862 3.689-13.862h5.587l3.743 13.862 3.953-13.862h6.589v.263l-7.168 23.193h-6.273l-3.743-13.811zm28.832 14.337c-5.693 0-8.012-2.268-8.012-7.59v-25.88h6.958v25.406c0 1.635.632 2.214 2.002 2.214.476 0 1.16-.157 1.582-.317h.105v5.746c-.579.21-1.581.421-2.635.421zm16.814 0c-7.273 0-12.228-4.586-12.228-12.122 0-7.328 4.849-12.389 11.279-12.389 7.01 0 10.489 4.692 10.489 11.703v2.636h-14.97c.843 3.636 3.689 4.583 6.536 4.583 2.478 0 4.27-.525 6.484-1.896h.264v5.429c-1.897 1.371-4.586 2.056-7.854 2.056zm-5.429-14.969h8.539c-.159-2.424-1.159-4.111-3.953-4.111-2.108-.001-3.954.895-4.586 4.111zm43.537-9.014h5.167v5.586h-5.167v9.542c0 2.055.791 3.004 2.899 3.004.58 0 1.318-.053 2.109-.317h.158v5.482c-.897.317-2.267.686-4.27.686-5.642 0-7.854-2.583-7.854-8.539v-9.857h-8.908v9.542c0 2.055.791 3.004 2.899 3.004.58 0 1.318-.053 2.108-.317h.159v5.482c-.896.317-2.267.686-4.27.686-5.641 0-7.854-2.583-7.854-8.539v-9.857h-3.585v-5.586h3.585v-6.43h6.958v6.43h8.908v-6.43h6.958v6.428zm41.261 1.687c0 7.169-4.797 11.07-12.073 11.07h-5.111v10.7h-7.166V127.77h12.277c7.277 0 12.073 3.9 12.073 11.174zm-12.652 5.008c3.639 0 5.378-2.056 5.378-5.008 0-3.057-1.739-5.112-5.378-5.112h-4.532v10.12h4.532zm28.147 14.442c-1.528 1.844-3.793 2.74-6.219 2.74-4.585 0-8.328-2.791-8.328-7.747 0-4.586 3.743-7.644 9.118-7.644 1.687 0 3.428.265 5.219.791v-.421c0-2.531-1.423-3.637-5.165-3.637-2.374 0-4.639.685-6.589 1.792h-.263v-5.587c1.791-1.054 4.954-1.951 8.062-1.951 7.064 0 10.911 3.374 10.911 9.278v14.706h-6.747v-2.32zm-.21-5.061v-2.003c-1.055-.579-2.423-.79-3.848-.79-2.267 0-3.636.843-3.636 2.74 0 1.952 1.369 2.795 3.425 2.795 1.95 0 3.426-1.003 4.059-2.742zm10.489-4.321c0-7.538 5.324-12.282 12.281-12.282 2.479 0 4.797.528 6.536 1.792v5.957h-.264c-1.529-1.16-3.321-1.846-5.376-1.846-3.479 0-6.114 2.373-6.114 6.378s2.636 6.325 6.114 6.325c2.055 0 3.847-.686 5.376-1.846h.264v5.956c-1.739 1.267-4.058 1.793-6.536 1.793-6.957.001-12.281-4.689-12.281-12.227zm30.147 1.055v10.646h-6.955V127.77h6.955v18.764l7.013-9.277h7.905v.263l-8.433 10.647 8.433 12.282v.264h-7.958l-6.96-10.646zm30.732 8.327c-1.528 1.844-3.796 2.74-6.221 2.74-4.586 0-8.328-2.791-8.328-7.747 0-4.586 3.742-7.644 9.12-7.644 1.687 0 3.426.265 5.218.791v-.421c0-2.531-1.424-3.637-5.165-3.637-2.371 0-4.641.685-6.591 1.792h-.262v-5.587c1.793-1.054 4.954-1.951 8.064-1.951 7.063 0 10.911 3.374 10.911 9.278v14.706h-6.747v-2.32zm-.211-5.061v-2.003c-1.054-.579-2.425-.79-3.849-.79-2.266 0-3.637.843-3.637 2.74 0 1.952 1.371 2.795 3.426 2.795 1.952 0 3.428-1.003 4.06-2.742zm18.66-12.281c1.263-2.583 3.108-4.059 5.693-4.059.947 0 1.896.21 2.263.421v6.642h-.262c-.79-.317-1.739-.528-3.058-.528-2.16 0-3.847 1.266-4.428 3.689v13.495h-6.957v-23.457h6.748v3.797zm26.614 17.237c-1.474 1.897-3.794 2.951-6.955 2.951-6.01 0-9.857-5.48-9.857-12.229 0-6.747 3.848-12.282 9.857-12.282 3.057 0 5.27.95 6.745 2.689V127.77h6.959v32.944h-6.749v-2.425zm-.21-5.535v-7.538c-1.159-1.687-2.686-2.424-4.427-2.424-3.056 0-5.111 2.214-5.111 6.22s2.056 6.167 5.111 6.167c1.742 0 3.268-.739 4.427-2.425zM48.492 171.471h20.346v4.005h-15.76v10.066h14.284v3.901H53.078v10.965h15.76v4.004H48.492v-32.941zm37.16 9.276c5.218 0 8.117 3.426 8.117 9.065v14.6h-4.374v-14.493c0-3.006-1.528-5.167-4.849-5.167-2.741 0-5.062 1.739-5.851 4.217v15.443H74.32v-23.19h4.375v3.372c1.369-2.16 3.689-3.847 6.957-3.847zm19.871.475h5.955v3.743h-5.955v12.49c0 2.636 1.37 3.532 3.847 3.532.685 0 1.423-.104 1.95-.315h.158v3.74c-.631.265-1.528.476-2.74.476-5.429 0-7.59-2.479-7.59-7.011v-12.912h-4.005v-3.743h4.005v-6.165h4.375v6.165zm20.187 23.666c-6.799 0-11.438-4.535-11.438-11.808 0-7.274 4.322-12.333 10.595-12.333 6.378 0 9.699 4.586 9.699 11.384v2.003h-15.918c.475 4.692 3.479 6.958 7.643 6.958 2.583 0 4.427-.579 6.483-2.108h.159v3.85c-1.898 1.475-4.376 2.054-7.223 2.054zm-6.904-14.339h11.543c-.157-3.424-1.845-6.059-5.429-6.059-3.321 0-5.482 2.476-6.114 6.059zm25.299-5.744c1.003-2.476 3.216-3.951 5.745-3.951 1.002 0 1.898.157 2.267.368v4.32h-.158c-.632-.315-1.687-.473-2.741-.473-2.372 0-4.374 1.581-5.113 4.217v15.126h-4.375v-23.19h4.375v3.583zm22.876-4.058c6.905 0 10.595 5.64 10.595 12.069 0 6.432-3.689 12.071-10.595 12.071-2.846 0-5.27-1.476-6.482-3.058v11.492h-4.375V181.22h4.375v2.634c1.212-1.631 3.636-3.107 6.482-3.107zm-.738 20.187c4.323 0 6.853-3.426 6.853-8.117 0-4.639-2.53-8.116-6.853-8.116-2.372 0-4.585 1.423-5.745 3.688v8.91c1.16 2.263 3.374 3.635 5.745 3.635zm20.874-16.129c1-2.476 3.215-3.951 5.745-3.951 1.001 0 1.897.157 2.266.368v4.32h-.158c-.632-.315-1.686-.473-2.741-.473-2.373 0-4.375 1.581-5.113 4.217v15.126h-4.375v-23.19h4.375v3.583zm14.336-12.965c1.528 0 2.794 1.211 2.794 2.741 0 1.527-1.266 2.74-2.794 2.74-1.476 0-2.793-1.213-2.793-2.74 0-1.53 1.318-2.741 2.793-2.741zm-2.161 9.382h4.375v23.19h-4.375v-23.19zm19.977 9.646c3.32 1.052 7.01 2.423 7.01 6.851 0 4.744-3.899 7.169-8.906 7.169-3.058 0-6.115-.739-7.854-2.108v-4.164h.21c1.951 1.793 4.849 2.583 7.59 2.583 2.478 0 4.691-.95 4.691-2.953 0-2.055-1.844-2.529-5.482-3.741-3.269-1.054-6.905-2.267-6.905-6.642 0-4.481 3.689-7.115 8.381-7.115 2.741 0 5.164.579 7.116 1.897v4.217h-.159c-1.896-1.528-4.111-2.425-6.852-2.425-2.742 0-4.27 1.212-4.27 2.847 0 1.843 1.686 2.37 5.43 3.584zm22.243 14.02c-6.8 0-11.438-4.535-11.438-11.808 0-7.274 4.322-12.333 10.594-12.333 6.379 0 9.699 4.586 9.699 11.384v2.003h-15.92c.475 4.692 3.48 6.958 7.644 6.958 2.584 0 4.428-.579 6.484-2.108h.157v3.85c-1.896 1.475-4.375 2.054-7.22 2.054zm-6.905-14.339h11.541c-.156-3.424-1.844-6.059-5.427-6.059-3.321 0-5.482 2.476-6.114 6.059z"/>\
</svg >'


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
