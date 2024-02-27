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
    return """
    #col_container {margin-left: auto; margin-right: auto; text-align: left;}

    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600&display=swap');
    
    body.dark{#warning {background-color: #555555};}
    
    #sidebar {
        order: 1;
        
        @media (max-width: 463px) {
          order: 2;
        }
    }
    
    #col-tabs {
        order: 2;
        
        @media (max-width: 463px) {
          order: 1;
        }
    }
    
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
    
    .gradio-container {
        max-width: none !important;
    }
    
    div.message {
        padding: var(--text-lg) !important;
    }
    
    div.message.user > div.icon-button {
        top: unset;
        bottom: 0;
    }
    
    div.message.bot > div.icon-button {
        top: unset;
        bottom: 0;
    }
    
    #prompt-form-row {
        position: relative;
    }
    
    #microphone-button {
        position: absolute;
        top: 14px;
        right: 125px;

        display: flex;
        justify-content: center;
        border: 1px solid var(--primary-500) !important;

        @media (max-width: 563px) {
          width: 20px;
        }
    }

    #microphone-button > img {
        margin-right: 0;
    }

    #add-button {
        position: absolute;
        top: 14px;
        right: 75px;
        
        display: flex;
        justify-content: center;
        border: 1px solid var(--primary-500) !important;
        
        @media (max-width: 563px) {
          width: 40px;
        }
    }
    
    #add-button > img {
        margin-right: 0;
    }

    #attach-button {
        position: absolute;
        top: 14px;
        right: 20px;
        
        display: flex;
        justify-content: center;
        border: 1px solid var(--primary-500) !important;
        
        @media (max-width: 563px) {
          width: 40px;
        }
    }
    
    #attach-button > img {
        margin-right: 40;
    }
    
    #prompt-form > label > textarea {
        padding-right: 0px;
        
        @media (max-width: 563px) {
          min-height: 94px;
          padding-right: 0px;
        }
    }

    #multi-selection > label > div.wrap > div.wrap-inner > div.secondary-wrap > div.remove-all {
        display: none !important;
    }
    
    #multi-selection > label > div.wrap > div.wrap-inner > div.token {
        display: none !important;
    }
    
    #multi-selection > label > div.wrap > div.wrap-inner > div.secondary-wrap::before {
        content: "Select_Any";
        padding: 0 4px;
        margin-right: 2px;
    }

    #single-selection > label > div.wrap > div.wrap-inner > div.secondary-wrap > div.remove-all {
        display: none !important;
    }

    #single-selection > label > div.wrap > div.wrap-inner > div.token {
        display: none !important;
    }

    #single-selection > label > div.wrap > div.wrap-inner > div.secondary-wrap::before {
        content: "Select_One";
        padding: 0 4px;
        margin-right: 2px;
    }

    #langchain_agents > label > div.wrap > div.wrap-inner > div.secondary-wrap > div.remove-all {
        display: none !important;
    }

    #langchain_agents > label > div.wrap > div.wrap-inner > div.token {
        display: none !important;
    }

    #langchain_agents > label > div.wrap > div.wrap-inner > div.secondary-wrap::before {
        content: "Select";
        padding: 0 4px;
        margin-right: 2px;
    }

 

#rating1, #rating2, #rating3, #rating4, #rating5 { /* Target all star buttons */ 
    all:unset ;
    font-size:2rem;
    display:flex ;
      width: 15px !important;      /* Set your desired width */
    padding-bottom: 15px !important; /* Set your desired

  transition: background-color 0.3s ease-in !important; 
  transition: color 0.3s ease-in !important; 
background-color: rgba(173, 181, 189, 0.5) !important;
clip-path: polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%);
}

    """
