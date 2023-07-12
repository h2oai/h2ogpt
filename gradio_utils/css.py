def get_css(kwargs) -> str:
    if kwargs['h2ocolors']:
        css_code = """footer {visibility: hidden;}
        body{background:linear-gradient(#f5f5f5,#e5e5e5);}
        body.dark{background:linear-gradient(#000000,#0d0d0d);}
        """
    else:
        css_code = """footer {visibility: hidden}"""

    css_code += make_css_base()
    return css_code


def make_css_base() -> str:
    css1 = """
        #col_container {margin-left: auto; margin-right: auto; text-align: left;}
        """
    return css1 + """
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600&display=swap');
    
    body.dark{#warning {background-color: #555555};}
    
    #small_btn {
        margin: 0.6em 0em 0.55em 0;
        max-width: 20em;
        min-width: 5em !important;
        height: 5em;
        font-size: 14px !important;
    }
    
    #prompt-form {
        border: 1px solid var(--primary-500) !important;
    }
    
    #prompt-form.block {
        border-radius: var(--block-radius) !important;
    }
    
    #prompt-form textarea {
        border: 1px solid rgb(209, 213, 219);
    }
    
    #prompt-form label > div {
        margin-top: 4px;
    }
    
    button.primary:hover {
        background-color: var(--primary-600) !important;
        transition: .2s;
    }
    
    #prompt-form-area {
        margin-bottom: 2.5rem;
    }
    .chatsmall chatbot {font-size: 10px !important}
    """
