from src.enums import split_google
from src.utils import sanitize_filename


def setup_app(name_login='google_login', name_app='h2ogpt', verbose=False):
    from authlib.integrations.starlette_client import OAuth, OAuthError
    from fastapi import FastAPI, Depends, Request
    from starlette.config import Config
    from starlette.responses import RedirectResponse
    from starlette.middleware.sessions import SessionMiddleware
    import os
    import gradio as gr

    assert os.environ['GOOGLE_CLIENT_ID'], "Set env GOOGLE_CLIENT_ID"
    GOOGLE_CLIENT_ID = os.environ['GOOGLE_CLIENT_ID']
    assert os.environ['GOOGLE_CLIENT_SECRET'], "Set env GOOGLE_CLIENT_SECRET"
    GOOGLE_CLIENT_SECRET = os.environ['GOOGLE_CLIENT_SECRET']
    assert os.environ['SECRET_KEY'], "Set env SECRET_KEY"
    SECRET_KEY = os.environ['SECRET_KEY']

    app = FastAPI()
    config = Config()
    oauth = OAuth(config)

    # Set up OAuth
    config_data = {'GOOGLE_CLIENT_ID': GOOGLE_CLIENT_ID, 'GOOGLE_CLIENT_SECRET': GOOGLE_CLIENT_SECRET}
    starlette_config = Config(environ=config_data)
    oauth = OAuth(starlette_config)
    oauth.register(
        name='google',
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'},
    )
    app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

    # Dependency to get the current user
    def get_user(request: Request):
        if verbose:
            print_request(request, which='get_user')
        user = request.session.get('user')
        if user:
            assert user['email'], "No email"
            assert user['email_verified'], "Email not verified: %s" % user['email']
            picture = user.get('picture', '') or 'None'
            return user['name'] + split_google + user['email'] + split_google + picture
        return None

    @app.get('/')
    def public(request: Request, user=Depends(get_user)):
        if verbose:
            print_request(request, which='public')
        root_url = gr.route_utils.get_root_url(request, "/", None)
        if user:
            return RedirectResponse(url=f'{root_url}/{name_app}/')
        else:
            return RedirectResponse(url=f'{root_url}/{name_login}/')

    @app.route('/logout')
    async def logout(request: Request):
        if verbose:
            print_request(request, which='logout')
        request.session.pop('user', None)
        return RedirectResponse(url='/')

    @app.route('/login')
    async def login(request: Request):
        if verbose:
            print_request(request, which='login0')
        root_url = gr.route_utils.get_root_url(request, "/login", None)
        redirect_uri = f"{root_url}/auth"
        print("Redirecting to", redirect_uri)
        return await oauth.google.authorize_redirect(request, redirect_uri)

    @app.route('/auth')
    async def auth(request: Request):
        if verbose:
            print_request(request, which='auth')
        try:
            access_token = await oauth.google.authorize_access_token(request)
        except OAuthError:
            print("Error getting access token", str(OAuthError))
            return RedirectResponse(url='/')
        request.session['user'] = dict(access_token)["userinfo"]
        print(f"Redirecting to /{name_app}")
        return RedirectResponse(url=f'/{name_app}')

    from urllib.parse import urlparse, urlunparse

    # Comment out below if using http instead of https
    @app.route('/login')
    async def login(request: Request):
        if verbose:
            print_request(request, which='login')
        parsed_url = urlparse(str(request.url_for('auth')))
        modified_url = parsed_url._replace(scheme='https')
        redirect_uri = urlunparse(modified_url)
        return await oauth.google.authorize_redirect(request, redirect_uri)

    def print_request(request: Request, which='unknown'):
        # Print request method (GET, POST, etc.)
        print("%s Method:" % which, request.method)

        # Print full URL
        print("%s URL:" % which, str(request.url))

        # Print headers
        print("%s Headers:" % which)
        for key, value in request.headers.items():
            print(f"    {key}: {value}")

        # Print query parameters
        print("%s Query Parameters:" % which)
        for key, value in request.query_params.items():
            print(f"    {key}: {value}")

        print("%s session:" % which, request.session)

    return app, get_user


def login_gradio(**kwargs):
    import gradio as gr
    login_demo = gr.Blocks()
    with login_demo:
        if kwargs['visible_h2ogpt_logo']:
            gr.Markdown(kwargs['markdown_logo'])
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=1):
                btn = gr.Button("%s Google Auth Login" % kwargs['page_title'])
            with gr.Column(scale=1):
                pass
        _js_redirect = """
            () => {
                url = '/login' + window.location.search;
                window.open(url, '_blank');
            }
            """
        btn.click(None, js=_js_redirect)
    return login_demo


def get_app(demo, app_kwargs={}, **login_kwargs):
    name_login = 'google_login'
    name_app = sanitize_filename(login_kwargs['page_title']).replace('/', '').lower()
    app, get_user = setup_app(name_login=name_login,
                              name_app=name_app,
                              verbose=False,  # can set to True to debug
                              )
    import gradio as gr
    login_app = gr.mount_gradio_app(app, login_gradio(**login_kwargs), f"/{name_login}")
    main_app = gr.mount_gradio_app(login_app, demo, path=f"/{name_app}",
                                   auth_dependency=get_user,
                                   app_kwargs=app_kwargs)
    return main_app
