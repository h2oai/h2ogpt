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

hpe_green = Color(
    name="green",
    c50="#f2fdfa",
    c100="#e6fbf5",
    c200="#b3f7e5",
    c300="#7cecd4",
    c400="#59e2c6",
    c500="#00b388",
    c600="#00a676",
    c700="#008a60",
    c800="#00734c",
    c900="#004d32",
    c950="#003221",
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
            primary_hue: colors.Color | str = hpe_green,
            secondary_hue: colors.Color | str = hpe_green,
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


# h2o_logo = '<svg xmlns = "http://www.w3.org/2000/svg" width = "25%" height = "25%" viewBox = "0 0 432 288" xml: space = "preserve" >\
# <path fill = "#00B388" d = "M48.492 73.322v32.943h114.691V73.322H48.492zm107.523 25.776H55.662V80.49h100.354v18.608z"/>\
# <path fill = "#FFF" d = "M55.661 160.714h-7.168V127.77h7.168v13.019H68.47V127.77h7.168v32.944H68.47v-13.652H55.661v13.652zm36.264.526c-7.273 0-12.229-4.586-12.229-12.122 0-7.328 4.85-12.389 11.28-12.389 7.01 0 10.488 4.692 10.488 11.703v2.636h-14.97c.844 3.636 3.69 4.583 6.537 4.583 2.477 0 4.269-.525 6.482-1.896h.264v5.429c-1.896 1.371-4.585 2.056-7.852 2.056zm-5.43-14.969h8.539c-.157-2.424-1.159-4.111-3.953-4.111-2.108-.001-3.953.895-4.586 4.111zm33.681.632-3.689 13.811h-6.272l-7.169-23.193v-.263h6.852l3.901 13.862 3.689-13.862h5.587l3.743 13.862 3.953-13.862h6.589v.263l-7.168 23.193h-6.273l-3.743-13.811zm28.832 14.337c-5.693 0-8.012-2.268-8.012-7.59v-25.88h6.958v25.406c0 1.635.632 2.214 2.002 2.214.476 0 1.16-.157 1.582-.317h.105v5.746c-.579.21-1.581.421-2.635.421zm16.814 0c-7.273 0-12.228-4.586-12.228-12.122 0-7.328 4.849-12.389 11.279-12.389 7.01 0 10.489 4.692 10.489 11.703v2.636h-14.97c.843 3.636 3.689 4.583 6.536 4.583 2.478 0 4.27-.525 6.484-1.896h.264v5.429c-1.897 1.371-4.586 2.056-7.854 2.056zm-5.429-14.969h8.539c-.159-2.424-1.159-4.111-3.953-4.111-2.108-.001-3.954.895-4.586 4.111zm43.537-9.014h5.167v5.586h-5.167v9.542c0 2.055.791 3.004 2.899 3.004.58 0 1.318-.053 2.109-.317h.158v5.482c-.897.317-2.267.686-4.27.686-5.642 0-7.854-2.583-7.854-8.539v-9.857h-8.908v9.542c0 2.055.791 3.004 2.899 3.004.58 0 1.318-.053 2.108-.317h.159v5.482c-.896.317-2.267.686-4.27.686-5.641 0-7.854-2.583-7.854-8.539v-9.857h-3.585v-5.586h3.585v-6.43h6.958v6.43h8.908v-6.43h6.958v6.428zm41.261 1.687c0 7.169-4.797 11.07-12.073 11.07h-5.111v10.7h-7.166V127.77h12.277c7.277 0 12.073 3.9 12.073 11.174zm-12.652 5.008c3.639 0 5.378-2.056 5.378-5.008 0-3.057-1.739-5.112-5.378-5.112h-4.532v10.12h4.532zm28.147 14.442c-1.528 1.844-3.793 2.74-6.219 2.74-4.585 0-8.328-2.791-8.328-7.747 0-4.586 3.743-7.644 9.118-7.644 1.687 0 3.428.265 5.219.791v-.421c0-2.531-1.423-3.637-5.165-3.637-2.374 0-4.639.685-6.589 1.792h-.263v-5.587c1.791-1.054 4.954-1.951 8.062-1.951 7.064 0 10.911 3.374 10.911 9.278v14.706h-6.747v-2.32zm-.21-5.061v-2.003c-1.055-.579-2.423-.79-3.848-.79-2.267 0-3.636.843-3.636 2.74 0 1.952 1.369 2.795 3.425 2.795 1.95 0 3.426-1.003 4.059-2.742zm10.489-4.321c0-7.538 5.324-12.282 12.281-12.282 2.479 0 4.797.528 6.536 1.792v5.957h-.264c-1.529-1.16-3.321-1.846-5.376-1.846-3.479 0-6.114 2.373-6.114 6.378s2.636 6.325 6.114 6.325c2.055 0 3.847-.686 5.376-1.846h.264v5.956c-1.739 1.267-4.058 1.793-6.536 1.793-6.957.001-12.281-4.689-12.281-12.227zm30.147 1.055v10.646h-6.955V127.77h6.955v18.764l7.013-9.277h7.905v.263l-8.433 10.647 8.433 12.282v.264h-7.958l-6.96-10.646zm30.732 8.327c-1.528 1.844-3.796 2.74-6.221 2.74-4.586 0-8.328-2.791-8.328-7.747 0-4.586 3.742-7.644 9.12-7.644 1.687 0 3.426.265 5.218.791v-.421c0-2.531-1.424-3.637-5.165-3.637-2.371 0-4.641.685-6.591 1.792h-.262v-5.587c1.793-1.054 4.954-1.951 8.064-1.951 7.063 0 10.911 3.374 10.911 9.278v14.706h-6.747v-2.32zm-.211-5.061v-2.003c-1.054-.579-2.425-.79-3.849-.79-2.266 0-3.637.843-3.637 2.74 0 1.952 1.371 2.795 3.426 2.795 1.952 0 3.428-1.003 4.06-2.742zm18.66-12.281c1.263-2.583 3.108-4.059 5.693-4.059.947 0 1.896.21 2.263.421v6.642h-.262c-.79-.317-1.739-.528-3.058-.528-2.16 0-3.847 1.266-4.428 3.689v13.495h-6.957v-23.457h6.748v3.797zm26.614 17.237c-1.474 1.897-3.794 2.951-6.955 2.951-6.01 0-9.857-5.48-9.857-12.229 0-6.747 3.848-12.282 9.857-12.282 3.057 0 5.27.95 6.745 2.689V127.77h6.959v32.944h-6.749v-2.425zm-.21-5.535v-7.538c-1.159-1.687-2.686-2.424-4.427-2.424-3.056 0-5.111 2.214-5.111 6.22s2.056 6.167 5.111 6.167c1.742 0 3.268-.739 4.427-2.425zM48.492 171.471h20.346v4.005h-15.76v10.066h14.284v3.901H53.078v10.965h15.76v4.004H48.492v-32.941zm37.16 9.276c5.218 0 8.117 3.426 8.117 9.065v14.6h-4.374v-14.493c0-3.006-1.528-5.167-4.849-5.167-2.741 0-5.062 1.739-5.851 4.217v15.443H74.32v-23.19h4.375v3.372c1.369-2.16 3.689-3.847 6.957-3.847zm19.871.475h5.955v3.743h-5.955v12.49c0 2.636 1.37 3.532 3.847 3.532.685 0 1.423-.104 1.95-.315h.158v3.74c-.631.265-1.528.476-2.74.476-5.429 0-7.59-2.479-7.59-7.011v-12.912h-4.005v-3.743h4.005v-6.165h4.375v6.165zm20.187 23.666c-6.799 0-11.438-4.535-11.438-11.808 0-7.274 4.322-12.333 10.595-12.333 6.378 0 9.699 4.586 9.699 11.384v2.003h-15.918c.475 4.692 3.479 6.958 7.643 6.958 2.583 0 4.427-.579 6.483-2.108h.159v3.85c-1.898 1.475-4.376 2.054-7.223 2.054zm-6.904-14.339h11.543c-.157-3.424-1.845-6.059-5.429-6.059-3.321 0-5.482 2.476-6.114 6.059zm25.299-5.744c1.003-2.476 3.216-3.951 5.745-3.951 1.002 0 1.898.157 2.267.368v4.32h-.158c-.632-.315-1.687-.473-2.741-.473-2.372 0-4.374 1.581-5.113 4.217v15.126h-4.375v-23.19h4.375v3.583zm22.876-4.058c6.905 0 10.595 5.64 10.595 12.069 0 6.432-3.689 12.071-10.595 12.071-2.846 0-5.27-1.476-6.482-3.058v11.492h-4.375V181.22h4.375v2.634c1.212-1.631 3.636-3.107 6.482-3.107zm-.738 20.187c4.323 0 6.853-3.426 6.853-8.117 0-4.639-2.53-8.116-6.853-8.116-2.372 0-4.585 1.423-5.745 3.688v8.91c1.16 2.263 3.374 3.635 5.745 3.635zm20.874-16.129c1-2.476 3.215-3.951 5.745-3.951 1.001 0 1.897.157 2.266.368v4.32h-.158c-.632-.315-1.686-.473-2.741-.473-2.373 0-4.375 1.581-5.113 4.217v15.126h-4.375v-23.19h4.375v3.583zm14.336-12.965c1.528 0 2.794 1.211 2.794 2.741 0 1.527-1.266 2.74-2.794 2.74-1.476 0-2.793-1.213-2.793-2.74 0-1.53 1.318-2.741 2.793-2.741zm-2.161 9.382h4.375v23.19h-4.375v-23.19zm19.977 9.646c3.32 1.052 7.01 2.423 7.01 6.851 0 4.744-3.899 7.169-8.906 7.169-3.058 0-6.115-.739-7.854-2.108v-4.164h.21c1.951 1.793 4.849 2.583 7.59 2.583 2.478 0 4.691-.95 4.691-2.953 0-2.055-1.844-2.529-5.482-3.741-3.269-1.054-6.905-2.267-6.905-6.642 0-4.481 3.689-7.115 8.381-7.115 2.741 0 5.164.579 7.116 1.897v4.217h-.159c-1.896-1.528-4.111-2.425-6.852-2.425-2.742 0-4.27 1.212-4.27 2.847 0 1.843 1.686 2.37 5.43 3.584zm22.243 14.02c-6.8 0-11.438-4.535-11.438-11.808 0-7.274 4.322-12.333 10.594-12.333 6.379 0 9.699 4.586 9.699 11.384v2.003h-15.92c.475 4.692 3.48 6.958 7.644 6.958 2.584 0 4.428-.579 6.484-2.108h.157v3.85c-1.896 1.475-4.375 2.054-7.22 2.054zm-6.905-14.339h11.541c-.156-3.424-1.844-6.059-5.427-6.059-3.321 0-5.482 2.476-6.114 6.059z"/>\
# </svg >'

hpe_logo = '<svg xmlns = "http://www.w3.org/2000/svg" width = "100%" height = "100%" viewBox = "0 0 432 288" xml: space = "preserve" >\
<path fill = "#00B388" d = "M48.492 73.322v32.943h114.691V73.322H48.492zm107.523 25.776H55.662V80.49h100.354v18.608z"/>\
<path fill = "#000000" d = "M55.661 160.714h-7.168V127.77h7.168v13.019H68.47V127.77h7.168v32.944H68.47v-13.652H55.661v13.652zm36.264.526c-7.273 0-12.229-4.586-12.229-12.122 0-7.328 4.85-12.389 11.28-12.389 7.01 0 10.488 4.692 10.488 11.703v2.636h-14.97c.844 3.636 3.69 4.583 6.537 4.583 2.477 0 4.269-.525 6.482-1.896h.264v5.429c-1.896 1.371-4.585 2.056-7.852 2.056zm-5.43-14.969h8.539c-.157-2.424-1.159-4.111-3.953-4.111-2.108-.001-3.953.895-4.586 4.111zm33.681.632-3.689 13.811h-6.272l-7.169-23.193v-.263h6.852l3.901 13.862 3.689-13.862h5.587l3.743 13.862 3.953-13.862h6.589v.263l-7.168 23.193h-6.273l-3.743-13.811zm28.832 14.337c-5.693 0-8.012-2.268-8.012-7.59v-25.88h6.958v25.406c0 1.635.632 2.214 2.002 2.214.476 0 1.16-.157 1.582-.317h.105v5.746c-.579.21-1.581.421-2.635.421zm16.814 0c-7.273 0-12.228-4.586-12.228-12.122 0-7.328 4.849-12.389 11.279-12.389 7.01 0 10.489 4.692 10.489 11.703v2.636h-14.97c.843 3.636 3.689 4.583 6.536 4.583 2.478 0 4.27-.525 6.484-1.896h.264v5.429c-1.897 1.371-4.586 2.056-7.854 2.056zm-5.429-14.969h8.539c-.159-2.424-1.159-4.111-3.953-4.111-2.108-.001-3.954.895-4.586 4.111zm43.537-9.014h5.167v5.586h-5.167v9.542c0 2.055.791 3.004 2.899 3.004.58 0 1.318-.053 2.109-.317h.158v5.482c-.897.317-2.267.686-4.27.686-5.642 0-7.854-2.583-7.854-8.539v-9.857h-8.908v9.542c0 2.055.791 3.004 2.899 3.004.58 0 1.318-.053 2.108-.317h.159v5.482c-.896.317-2.267.686-4.27.686-5.641 0-7.854-2.583-7.854-8.539v-9.857h-3.585v-5.586h3.585v-6.43h6.958v6.43h8.908v-6.43h6.958v6.428zm41.261 1.687c0 7.169-4.797 11.07-12.073 11.07h-5.111v10.7h-7.166V127.77h12.277c7.277 0 12.073 3.9 12.073 11.174zm-12.652 5.008c3.639 0 5.378-2.056 5.378-5.008 0-3.057-1.739-5.112-5.378-5.112h-4.532v10.12h4.532zm28.147 14.442c-1.528 1.844-3.793 2.74-6.219 2.74-4.585 0-8.328-2.791-8.328-7.747 0-4.586 3.743-7.644 9.118-7.644 1.687 0 3.428.265 5.219.791v-.421c0-2.531-1.423-3.637-5.165-3.637-2.374 0-4.639.685-6.589 1.792h-.263v-5.587c1.791-1.054 4.954-1.951 8.062-1.951 7.064 0 10.911 3.374 10.911 9.278v14.706h-6.747v-2.32zm-.21-5.061v-2.003c-1.055-.579-2.423-.79-3.848-.79-2.267 0-3.636.843-3.636 2.74 0 1.952 1.369 2.795 3.425 2.795 1.95 0 3.426-1.003 4.059-2.742zm10.489-4.321c0-7.538 5.324-12.282 12.281-12.282 2.479 0 4.797.528 6.536 1.792v5.957h-.264c-1.529-1.16-3.321-1.846-5.376-1.846-3.479 0-6.114 2.373-6.114 6.378s2.636 6.325 6.114 6.325c2.055 0 3.847-.686 5.376-1.846h.264v5.956c-1.739 1.267-4.058 1.793-6.536 1.793-6.957.001-12.281-4.689-12.281-12.227zm30.147 1.055v10.646h-6.955V127.77h6.955v18.764l7.013-9.277h7.905v.263l-8.433 10.647 8.433 12.282v.264h-7.958l-6.96-10.646zm30.732 8.327c-1.528 1.844-3.796 2.74-6.221 2.74-4.586 0-8.328-2.791-8.328-7.747 0-4.586 3.742-7.644 9.12-7.644 1.687 0 3.426.265 5.218.791v-.421c0-2.531-1.424-3.637-5.165-3.637-2.371 0-4.641.685-6.591 1.792h-.262v-5.587c1.793-1.054 4.954-1.951 8.064-1.951 7.063 0 10.911 3.374 10.911 9.278v14.706h-6.747v-2.32zm-.211-5.061v-2.003c-1.054-.579-2.425-.79-3.849-.79-2.266 0-3.637.843-3.637 2.74 0 1.952 1.371 2.795 3.426 2.795 1.952 0 3.428-1.003 4.06-2.742zm18.66-12.281c1.263-2.583 3.108-4.059 5.693-4.059.947 0 1.896.21 2.263.421v6.642h-.262c-.79-.317-1.739-.528-3.058-.528-2.16 0-3.847 1.266-4.428 3.689v13.495h-6.957v-23.457h6.748v3.797zm26.614 17.237c-1.474 1.897-3.794 2.951-6.955 2.951-6.01 0-9.857-5.48-9.857-12.229 0-6.747 3.848-12.282 9.857-12.282 3.057 0 5.27.95 6.745 2.689V127.77h6.959v32.944h-6.749v-2.425zm-.21-5.535v-7.538c-1.159-1.687-2.686-2.424-4.427-2.424-3.056 0-5.111 2.214-5.111 6.22s2.056 6.167 5.111 6.167c1.742 0 3.268-.739 4.427-2.425zM48.492 171.471h20.346v4.005h-15.76v10.066h14.284v3.901H53.078v10.965h15.76v4.004H48.492v-32.941zm37.16 9.276c5.218 0 8.117 3.426 8.117 9.065v14.6h-4.374v-14.493c0-3.006-1.528-5.167-4.849-5.167-2.741 0-5.062 1.739-5.851 4.217v15.443H74.32v-23.19h4.375v3.372c1.369-2.16 3.689-3.847 6.957-3.847zm19.871.475h5.955v3.743h-5.955v12.49c0 2.636 1.37 3.532 3.847 3.532.685 0 1.423-.104 1.95-.315h.158v3.74c-.631.265-1.528.476-2.74.476-5.429 0-7.59-2.479-7.59-7.011v-12.912h-4.005v-3.743h4.005v-6.165h4.375v6.165zm20.187 23.666c-6.799 0-11.438-4.535-11.438-11.808 0-7.274 4.322-12.333 10.595-12.333 6.378 0 9.699 4.586 9.699 11.384v2.003h-15.918c.475 4.692 3.479 6.958 7.643 6.958 2.583 0 4.427-.579 6.483-2.108h.159v3.85c-1.898 1.475-4.376 2.054-7.223 2.054zm-6.904-14.339h11.543c-.157-3.424-1.845-6.059-5.429-6.059-3.321 0-5.482 2.476-6.114 6.059zm25.299-5.744c1.003-2.476 3.216-3.951 5.745-3.951 1.002 0 1.898.157 2.267.368v4.32h-.158c-.632-.315-1.687-.473-2.741-.473-2.372 0-4.374 1.581-5.113 4.217v15.126h-4.375v-23.19h4.375v3.583zm22.876-4.058c6.905 0 10.595 5.64 10.595 12.069 0 6.432-3.689 12.071-10.595 12.071-2.846 0-5.27-1.476-6.482-3.058v11.492h-4.375V181.22h4.375v2.634c1.212-1.631 3.636-3.107 6.482-3.107zm-.738 20.187c4.323 0 6.853-3.426 6.853-8.117 0-4.639-2.53-8.116-6.853-8.116-2.372 0-4.585 1.423-5.745 3.688v8.91c1.16 2.263 3.374 3.635 5.745 3.635zm20.874-16.129c1-2.476 3.215-3.951 5.745-3.951 1.001 0 1.897.157 2.266.368v4.32h-.158c-.632-.315-1.686-.473-2.741-.473-2.373 0-4.375 1.581-5.113 4.217v15.126h-4.375v-23.19h4.375v3.583zm14.336-12.965c1.528 0 2.794 1.211 2.794 2.741 0 1.527-1.266 2.74-2.794 2.74-1.476 0-2.793-1.213-2.793-2.74 0-1.53 1.318-2.741 2.793-2.741zm-2.161 9.382h4.375v23.19h-4.375v-23.19zm19.977 9.646c3.32 1.052 7.01 2.423 7.01 6.851 0 4.744-3.899 7.169-8.906 7.169-3.058 0-6.115-.739-7.854-2.108v-4.164h.21c1.951 1.793 4.849 2.583 7.59 2.583 2.478 0 4.691-.95 4.691-2.953 0-2.055-1.844-2.529-5.482-3.741-3.269-1.054-6.905-2.267-6.905-6.642 0-4.481 3.689-7.115 8.381-7.115 2.741 0 5.164.579 7.116 1.897v4.217h-.159c-1.896-1.528-4.111-2.425-6.852-2.425-2.742 0-4.27 1.212-4.27 2.847 0 1.843 1.686 2.37 5.43 3.584zm22.243 14.02c-6.8 0-11.438-4.535-11.438-11.808 0-7.274 4.322-12.333 10.594-12.333 6.379 0 9.699 4.586 9.699 11.384v2.003h-15.92c.475 4.692 3.48 6.958 7.644 6.958 2.584 0 4.428-.579 6.484-2.108h.157v3.85c-1.896 1.475-4.375 2.054-7.22 2.054zm-6.905-14.339h11.541c-.156-3.424-1.844-6.059-5.427-6.059-3.321 0-5.482 2.476-6.114 6.059z"/>\
</svg >'

chatmate_logo = '<svg version = "1.0" xmlns = "http://www.w3.org/2000/svg" width = "100%" height = "100%" viewBox = "0 0 432 288" preserveAspectRatio = "xMidYMid meet" >\
<g transform = "translate(0.000000,78.000000) scale(0.100000,-0.100000)" fill = "#00b388" stroke = "none" >\
<path d ="M45 693 c-45 - 12 - 45 - 12 - 45 - 303 0 - 245 2 - 279 16 - 291 10 - 8 48\
-15 89 - 17 l72 - 4 17 - 39 17 - 39 91 0 90 0 15 40 16 40 77 0 c104 0 140 - 8\
140 - 30 0 - 46 8 - 50 102 - 50 l90 0 49 41 49 41 439 - 1 c385 - 2 444 0 478 14\
l38 16 51 - 55 51 - 56 80 0 c76 0 80 1 91 25 7 14 12 32 12 39 0 16 152 26 276\
18 l81 - 5 17 - 38 17 - 39 91 0 90 0 15 40 16 40 134 0 c99 0 143 4 171 16 l37\
15 51 - 55 51 - 56 80 0 c76 0 80 1 91 25 7 14 12 32 12 40 0 17 51 20 345 15\
154 - 2 197 1 228 13 l37 16 0 281 0 281 - 36 15 c-30 13 - 88 14 - 372 10 - 186\
-3 - 355 - 5 - 377 - 5 - 47 1 - 200 2 - 265 1 - 25 0 - 91 1 - 147 1 - 91 2 - 104 0 - 132\
-21 l-31 - 23 - 31 23 c-30 23 - 35 23 - 292 22 - 144 - 1 - 284 - 1 - 312 - 2 - 27 0\
-93 0 - 145 0 - 52 0 - 117 0 - 145 0 - 27 0 - 345 1 - 707 1 l-656 2 - 31 - 23 - 31\
-23 - 31 23 c-26 19 - 44 23 - 117 25 - 48 1 - 96 - 1 - 107 - 4z m201 - 178 c24 - 44\
47 - 81 53 - 83 5 - 2 30 36 55 83 l47 85 65 0 64 0 0 - 210 0 - 210 - 62 0 - 63 0 3\
75 c2 41 0 75 - 5 75 - 5 0 - 29 - 58 - 53 - 130 - 25 - 71 - 47 - 130 - 50 - 130 - 3 0\
-26 60 - 50 133 l-45 132 - 3 - 77 - 3 - 78 - 64 0 - 65 0 0 210 0 211 66 - 3 67 - 3\
43 - 80z m454 - 65 l0 - 150 35 0 35 0 0 150 0 150 60 0 60 0 0 - 196 0 - 196 - 72\
-74 c-39 - 41 - 73 - 72 - 75 - 71 - 2 2 4 26 12 53 8 27 15 52 15 57 0 4 - 36 7 - 80\
7 - 80 0 - 80 0 - 100 33 - 18 29 - 20 50 - 20 210 l0 177 65 0 65 0 0 - 150z m570\
85 l0 - 64 - 102 - 3 - 103 - 3 0 - 80 0 - 80 103 - 3 102 - 3 0 - 59 0 - 60 - 139 0\
c-125 0 - 141 2 - 165 21 l-26 20 0 156 0 155 34 34 34 34 131 0 131 0 0 - 65z\
m180 - 25 l0 - 90 35 0 35 0 0 90 0 90 60 0 60 0 0 - 210 0 - 210 - 60 0 - 60 0 0\
90 0 90 - 35 0 - 35 0 0 - 90 0 - 90 - 65 0 - 65 0 0 210 0 210 65 0 65 0 0 - 90z\
m552 - 175 c37 - 143 67 - 262 68 - 263 0 - 2 - 33 35 - 72 82 l-73 85 - 51 1 c-47 0\
-52 - 2 - 58 - 25 - 6 - 23 - 11 - 25 - 66 - 25 - 33 0 - 60 4 - 60 10 0 5 23 96 50 201\
28 105 50 194 50 196 0 2 32 3 72 1 l72 - 3 68 - 260z m368 200 l0 - 65 - 35 0\
-35 0 0 - 145 0 - 145 - 70 0 - 70 0 0 145 0 145 - 40 0 - 40 0 0 65 0 65 145 0 145\
0 0 - 65z m226 - 20 c24 - 44 47 - 81 53 - 83 5 - 2 30 36 55 83 l47 85 65 0 64 0 0\
-210 0 - 210 - 62 0 - 63 0 3 75 c2 41 0 75 - 5 75 - 5 0 - 29 - 58 - 53 - 130 - 25 - 71\
-47 - 130 - 50 - 130 - 3 0 - 26 60 - 50 133 l-45 132 - 3 - 77 - 3 - 78 - 64 0 - 65 0 0\
210 0 211 66 - 3 67 - 3 43 - 80z m636 - 180 c37 - 143 67 - 262 68 - 263 0 - 2 - 33\
35 - 72 82 l-73 85 - 51 1 c-47 0 - 52 - 2 - 58 - 25 - 6 - 23 - 11 - 25 - 66 - 25 - 33 0\
-60 4 - 60 10 0 5 23 96 50 201 28 105 50 194 50 196 0 2 32 3 72 1 l72 - 3 68\
-260z m368 200 l0 - 65 - 35 0 - 35 0 0 - 145 0 - 145 - 70 0 - 70 0 0 145 0 145 - 40\
0 - 40 0 0 65 0 65 145 0 145 0 0 - 65z m337 5 l0 - 60 - 79 0 - 78 0 0 - 30 0 - 30\
70 0 70 0 0 - 30 0 - 30 - 70 0 - 70 0 0 - 30 0 - 30 78 0 79 0 0 - 60 0 - 60 - 144 0\
-143 0 0 210 0 210 143 0 144 0 0 - 60z"/ > </g > </svg >'

# hpe_logo = '<svg version="1.1" id="primary_logo" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" width="25%" height="25%" viewBox="24.246 36.661 167.582 69.999" enable-background="new 0 0 600.28 600.28" xml:space="preserve"><g><path id="element" fill="#00B388" d="M24.246 36.661v16.472H81.592V36.661H24.246zm53.762 12.888H27.831V40.245h50.177v9.304z"/><path fill="#000000" d="M27.831 80.357h-3.584V63.885h3.584v6.509h6.404v-6.509h3.584v16.472h-3.584v-6.826h-6.404v6.826zm18.132 0.263c-3.637 0 -6.114 -2.293 -6.114 -6.061 0 -3.664 2.425 -6.194 5.64 -6.194 3.505 0 5.244 2.346 5.244 5.852v1.318h-7.485c0.422 1.818 1.845 2.292 3.268 2.292 1.238 0 2.134 -0.263 3.241 -0.948h0.132v2.714c-0.948 0.686 -2.292 1.028 -3.926 1.028zm-2.715 -7.484h4.269c-0.078 -1.212 -0.579 -2.056 -1.977 -2.056 -1.054 -0.001 -1.977 0.448 -2.293 2.056zm16.841 0.316 -1.844 6.906H55.108l-3.584 -11.597v-0.132h3.426l1.951 6.931 1.844 -6.931h2.793l1.872 6.931 1.977 -6.931h3.294v0.132l-3.584 11.597h-3.137l-1.872 -6.906zm14.416 7.168c-2.847 0 -4.006 -1.134 -4.006 -3.795V63.885h3.479v12.703c0 0.818 0.316 1.107 1.001 1.107 0.238 0 0.58 -0.078 0.791 -0.158h0.053v2.873a4.167 4.167 0 0 1 -1.318 0.211zm8.407 0c-3.637 0 -6.114 -2.293 -6.114 -6.061 0 -3.664 2.424 -6.194 5.639 -6.194 3.505 0 5.244 2.346 5.244 5.852v1.318h-7.485c0.422 1.818 1.844 2.292 3.268 2.292 1.239 0 2.135 -0.263 3.242 -0.948h0.132v2.714c-0.948 0.686 -2.293 1.028 -3.927 1.028zm-2.714 -7.484h4.269c-0.079 -1.212 -0.579 -2.056 -1.977 -2.056 -1.054 -0.001 -1.977 0.448 -2.293 2.056zm21.768 -4.507h2.583v2.793h-2.583v4.771c0 1.028 0.396 1.502 1.449 1.502 0.29 0 0.659 -0.027 1.054 -0.158h0.079v2.741c-0.448 0.158 -1.133 0.343 -2.135 0.343 -2.821 0 -3.927 -1.292 -3.927 -4.269v-4.928h-4.454v4.771c0 1.028 0.396 1.502 1.449 1.502 0.29 0 0.659 -0.027 1.054 -0.158h0.079v2.741c-0.448 0.158 -1.133 0.343 -2.135 0.343 -2.821 0 -3.927 -1.292 -3.927 -4.269v-4.928h-1.792v-2.793h1.792v-3.215h3.479v3.215h4.454v-3.215h3.479v3.214zm20.631 0.843c0 3.584 -2.398 5.535 -6.037 5.535h-2.556v5.35h-3.583V63.885h6.138c3.638 0 6.037 1.95 6.037 5.587zm-6.326 2.504c1.819 0 2.689 -1.028 2.689 -2.504 0 -1.528 -0.869 -2.556 -2.689 -2.556h-2.266v5.06h2.266zm14.073 7.221c-0.764 0.922 -1.897 1.37 -3.109 1.37 -2.292 0 -4.164 -1.396 -4.164 -3.873 0 -2.293 1.872 -3.822 4.559 -3.822 0.843 0 1.714 0.133 2.609 0.396v-0.211c0 -1.266 -0.712 -1.818 -2.583 -1.818 -1.187 0 -2.319 0.342 -3.294 0.896h-0.132V69.342c0.896 -0.527 2.477 -0.976 4.031 -0.976 3.532 0 5.456 1.687 5.456 4.639v7.353h-3.373v-1.16zM130.238 76.667v-1.002c-0.528 -0.289 -1.212 -0.395 -1.924 -0.395 -1.133 0 -1.818 0.422 -1.818 1.37 0 0.976 0.684 1.398 1.713 1.398 0.975 0 1.713 -0.502 2.029 -1.371zm5.244 -2.161c0 -3.769 2.662 -6.141 6.141 -6.141 1.239 0 2.398 0.264 3.268 0.896v2.978h-0.132c-0.764 -0.58 -1.661 -0.923 -2.688 -0.923 -1.739 0 -3.057 1.187 -3.057 3.189s1.318 3.163 3.057 3.163c1.028 0 1.923 -0.343 2.688 -0.923h0.132v2.978c-0.869 0.633 -2.029 0.897 -3.268 0.897 -3.478 0.001 -6.141 -2.344 -6.141 -6.113zm15.073 0.528v5.323h-3.478V63.885h3.478V73.267l3.507 -4.638h3.953v0.132l-4.217 5.323 4.217 6.141v0.132h-3.979l-3.48 -5.323zm15.366 4.163c-0.764 0.922 -1.898 1.37 -3.111 1.37 -2.293 0 -4.164 -1.396 -4.164 -3.873 0 -2.293 1.871 -3.822 4.56 -3.822 0.843 0 1.713 0.133 2.609 0.396v-0.211c0 -1.266 -0.712 -1.818 -2.583 -1.818 -1.186 0 -2.321 0.342 -3.296 0.896h-0.131v-2.793c0.897 -0.527 2.477 -0.976 4.032 -0.976 3.532 0 5.456 1.687 5.456 4.639v7.353h-3.373v-1.16zm-0.106 -2.531v-1.002c-0.527 -0.289 -1.213 -0.395 -1.924 -0.395 -1.133 0 -1.818 0.422 -1.818 1.37 0 0.976 0.686 1.398 1.713 1.398 0.976 0 1.714 -0.502 2.03 -1.371zm9.33 -6.141c0.632 -1.292 1.554 -2.029 2.847 -2.029 0.473 0 0.948 0.105 1.132 0.211v3.321h-0.131c-0.395 -0.158 -0.869 -0.264 -1.529 -0.264 -1.08 0 -1.923 0.633 -2.214 1.844v6.747h-3.478V68.627h3.374v1.898zm13.307 8.618c-0.737 0.948 -1.897 1.476 -3.478 1.476 -3.005 0 -4.928 -2.74 -4.928 -6.114 0 -3.373 1.924 -6.141 4.928 -6.141 1.528 0 2.635 0.475 3.373 1.344v-5.823h3.479v16.472h-3.374v-1.213zm-0.105 -2.768v-3.769c-0.579 -0.843 -1.343 -1.212 -2.213 -1.212 -1.528 0 -2.556 1.107 -2.556 3.11s1.028 3.083 2.556 3.083c0.871 0 1.634 -0.369 2.213 -1.213zM24.246 85.736h10.173v2.002h-7.88v5.033h7.142v1.951h-7.142v5.482h7.88v2.002H24.246v-16.471zm18.58 4.638c2.609 0 4.058 1.713 4.058 4.533v7.3h-2.187v-7.247c0 -1.503 -0.764 -2.583 -2.424 -2.583 -1.371 0 -2.531 0.869 -2.926 2.108v7.722h-2.188v-11.595h2.188v1.686c0.684 -1.08 1.844 -1.923 3.478 -1.923zm9.936 0.237h2.978v1.872h-2.978v6.245c0 1.318 0.685 1.766 1.923 1.766 0.342 0 0.712 -0.052 0.975 -0.158h0.079v1.87c-0.316 0.133 -0.764 0.238 -1.37 0.238 -2.714 0 -3.795 -1.239 -3.795 -3.506v-6.456h-2.002v-1.872h2.002v-3.083h2.188v3.083zm10.093 11.833c-3.399 0 -5.719 -2.268 -5.719 -5.904s2.161 -6.167 5.298 -6.167c3.189 0 4.849 2.293 4.849 5.692v1.002h-7.959c0.237 2.346 1.739 3.479 3.822 3.479 1.292 0 2.213 -0.289 3.242 -1.054h0.079v1.925c-0.949 0.738 -2.188 1.027 -3.612 1.027zm-3.452 -7.169h5.772c-0.078 -1.712 -0.922 -3.029 -2.714 -3.029 -1.661 0 -2.741 1.238 -3.057 3.029zm12.649 -2.872c0.502 -1.238 1.608 -1.976 2.873 -1.976 0.501 0 0.949 0.078 1.133 0.184v2.16h-0.079c-0.316 -0.158 -0.843 -0.237 -1.371 -0.237 -1.186 0 -2.187 0.791 -2.557 2.108v7.563h-2.188v-11.595h2.188v1.792zm11.438 -2.029c3.453 0 5.298 2.82 5.298 6.034 0 3.216 -1.844 6.036 -5.298 6.036 -1.423 0 -2.635 -0.738 -3.241 -1.529v5.746h-2.188v-16.052H80.25v1.317c0.606 -0.816 1.818 -1.553 3.241 -1.553zm-0.369 10.093c2.162 0 3.427 -1.713 3.427 -4.058 0 -2.319 -1.265 -4.058 -3.427 -4.058 -1.186 0 -2.292 0.712 -2.873 1.844v4.455c0.58 1.132 1.687 1.818 2.873 1.818zm10.437 -8.064c0.5 -1.238 1.608 -1.976 2.873 -1.976 0.501 0 0.948 0.078 1.133 0.184v2.16h-0.079c-0.316 -0.158 -0.843 -0.237 -1.371 -0.237 -1.187 0 -2.188 0.791 -2.557 2.108v7.563h-2.188v-11.595h2.188v1.792zm7.168 -6.482c0.764 0 1.397 0.606 1.397 1.371 0 0.763 -0.633 1.37 -1.397 1.37 -0.738 0 -1.397 -0.607 -1.397 -1.37 0 -0.765 0.659 -1.371 1.397 -1.371zm-1.081 4.691h2.188v11.595h-2.188v-11.595zm9.988 4.823c1.66 0.526 3.505 1.212 3.505 3.426 0 2.372 -1.949 3.584 -4.453 3.584 -1.529 0 -3.058 -0.369 -3.927 -1.054v-2.082h0.105c0.976 0.897 2.424 1.292 3.795 1.292 1.239 0 2.346 -0.475 2.346 -1.477 0 -1.028 -0.922 -1.264 -2.741 -1.871 -1.634 -0.527 -3.453 -1.133 -3.453 -3.321 0 -2.241 1.844 -3.558 4.191 -3.558 1.371 0 2.582 0.289 3.558 0.948v2.108h-0.079c-0.948 -0.764 -2.056 -1.213 -3.426 -1.213 -1.371 0 -2.135 0.606 -2.135 1.423 0 0.922 0.843 1.185 2.715 1.792zm11.122 7.01c-3.4 0 -5.719 -2.268 -5.719 -5.904s2.161 -6.167 5.297 -6.167c3.189 0 4.849 2.293 4.849 5.692v1.002h-7.96c0.237 2.346 1.74 3.479 3.822 3.479 1.292 0 2.214 -0.289 3.242 -1.054h0.078v1.925c-0.948 0.738 -2.188 1.027 -3.61 1.027zm-3.453 -7.169h5.771c-0.078 -1.712 -0.922 -3.029 -2.713 -3.029 -1.661 0 -2.741 1.238 -3.057 3.029z"/></g></svg>'


# def get_h2o_title(title, description):
#     # NOTE: Check full width desktop, smallest width browser desktop, iPhone browsers to ensure no overlap etc.
#     return f"""<div style="float:left; justify-content:left; height: 80px; width: 195px; margin-top:0px">
#                     {description}
#                 </div>
#                 <div style="display:flex; justify-content:center; margin-bottom:30px; margin-right:330px;">
#                     <div style="height: 60px; width: 60px; margin-right:20px;">{hpe_logo}</div>
#                     <h1 style="line-height:60px">{title}</h1>
#                 </div>
#                 <div style="float:right; height: 80px; width: 80px; margin-top:-100px">
#                     <img src="https://raw.githubusercontent.com/h2oai/h2ogpt/main/docs/h2o-qr.png">
#                 </div>
#                 """


def get_h2o_title(title, description):
    # NOTE: Check full width desktop, smallest width browser desktop, iPhone browsers to ensure no overlap etc.
    return f"""
                <div style="display:flex; float:center; justify-content:center; margin-bottom:30px; margin-right:50px;">
                    <div style="height: 70px; width: 70px; margin-right:20px;">{hpe_logo}</div>
                    <h1 style="line-height:60px">{title}</h1>
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
