import json

import uvicorn
from authlib.integrations.base_client import OAuthError
from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.requests import Request
import gradio as gr
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config

app = FastAPI()

import os

assert os.environ['GOOGLE_CLIENT_ID'], "Set env GOOGLE_CLIENT_ID"
assert os.environ['GOOGLE_CLIENT_SECRET'], "Set env GOOGLE_CLIENT_SECRET"

config = Config()
oauth = OAuth(config)

CONF_URL = 'https://accounts.google.com/.well-known/openid-configuration'
oauth.register(
    name='google',
    server_metadata_url=CONF_URL,
    client_kwargs={
        'scope': 'openid email profile'
    }
)

from urllib.parse import urlparse, urlunparse


# The Middleware that enforces authentication on /gradio app
@app.middleware("http")
async def check_authentication(request: Request, call_next):
    if request.url.path.startswith('/login') or request.url.path.startswith('/auth'):
        # Skip authentication check for login and authentication routes
        return await call_next(request)

    if request.url.path == '/gradio/api/predict' or request.url.path == '/gradio/reset':
        return await call_next(request)

    user = request.session.get("user")
    if not user:
        # User is not logged in, redirect to login page
        return RedirectResponse(url="/login")

    return await call_next(request)


@app.get('/')
async def homepage(request: Request):
    user = request.session.get('user')
    if user:
        data = json.dumps(user)
        html = (
            f'<pre>{data}</pre>'
            '<a href="/logout">logout</a>'
            '<br>'
            '<a href="/gradio">demo</a>'
        )
        return HTMLResponse(html)
    return HTMLResponse('<a href="/login">login</a>')


@app.get('/login')
async def login(request: Request):
    redirect_uri = request.url_for('auth')
    return await oauth.google.authorize_redirect(request, redirect_uri)

# If using http and not https, then comment out this function:
@app.route('/login')
async def login(request: Request):
    parsed_url = urlparse(str(request.url_for('auth')))
    modified_url = parsed_url._replace(scheme='https')
    redirect_uri = urlunparse(modified_url)
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get('/auth')
async def auth(request: Request):
    print(f"before request user {request.session.get('user')}")
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as error:
        return HTMLResponse(f'<h1>{error.error}</h1>')
    user = token.get('userinfo')
    if user:
        request.session['user'] = dict(user)
    print(f"after request user {request.session.get('user')}")
    return RedirectResponse(url='/')


@app.get('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
    return RedirectResponse(url='/')


# CODE FOR MOUNTED GRADIO APP

def update(name, request: gr.Request):
    return f"Welcome to Gradio, {name}!\n{request.request.session.get('user')}"


def make_demo_visible(request: gr.Request):
    if request.request.session.get('user'):
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
        value="Looks like you are not logged in. Please login at the main app.")


def get_demo():
    with gr.Blocks() as demo:
        start_btn = gr.Button("Press Here to initialize the demo!")

        with gr.Row():
            inp = gr.Textbox(placeholder="What is your name?", visible=False)
            out = gr.Textbox(visible=False)

        btn = gr.Button("Run", visible=False)

        start_btn.click(make_demo_visible, outputs=[inp, out, btn, start_btn])
        btn.click(fn=update, inputs=inp, outputs=out)
    return demo


def get_app(demo, app_kwargs):
    gradio_app = gr.mount_gradio_app(app, demo, "/gradio", app_kwargs)
    app.add_middleware(SessionMiddleware, secret_key="!secret")
    return gradio_app
