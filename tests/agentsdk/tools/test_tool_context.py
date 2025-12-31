# -*- coding: utf-8 -*-
import pytest

from agentsdk.tools.tool_context import ToolContext, AuthType, AuthConfig, AuthCredential, HttpCredentials


@pytest.fixture
def ctx():
    return ToolContext(function_call_id="test_call", state={"user": "alice"})

def test_initial_state(ctx):
    assert ctx.function_call_id == "test_call"
    assert ctx.state["user"] == "alice"
    assert ctx._credentials == {}

def test_set_and_get_auth(ctx):
    auth_cfg = AuthConfig(scheme=AuthType.API_KEY, key="my_key")
    credential = AuthCredential(auth_type=AuthType.API_KEY, api_key="123456")
    ctx.set_auth(auth_cfg, credential)
    retrieved = ctx.get_auth(auth_cfg)
    assert retrieved.api_key == "123456"
    assert retrieved.auth_type == AuthType.API_KEY

def test_require_auth_existing(ctx):
    auth_cfg = AuthConfig(scheme=AuthType.HTTP, key="http_key")
    credential = AuthCredential(auth_type=AuthType.HTTP, http=HttpCredentials(username="user", password="pass"))
    ctx.set_auth(auth_cfg, credential)
    result = ctx.require_auth(auth_cfg)
    assert result.auth_type == AuthType.HTTP
    assert result.http.username == "user"
    assert result.http.password == "pass"

def test_multiple_credentials(ctx):
    cfg1 = AuthConfig(scheme=AuthType.API_KEY, key="key1")
    cred1 = AuthCredential(auth_type=AuthType.API_KEY, api_key="a1")
    cfg2 = AuthConfig(scheme=AuthType.HTTP, key="key2")
    cred2 = AuthCredential(auth_type=AuthType.HTTP, http=HttpCredentials(username="u2", password="p2"))
    
    ctx.set_auth(cfg1, cred1)
    ctx.set_auth(cfg2, cred2)

    r1 = ctx.get_auth(cfg1)
    r2 = ctx.get_auth(cfg2)
    assert r1.api_key == "a1"
    assert r2.http.username == "u2"
