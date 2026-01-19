# -*- coding: utf-8 -*-
"""
This file provides the tool context implementation.
"""

from enum import Enum
from typing import Any, Optional, Dict

from pydantic import BaseModel

State = Any


class AuthType(str, Enum):
    API_KEY = "apiKey"
    HTTP = "http"
    SERVICE_ACCOUNT = "serviceAccount"

class HttpCredentials(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None


class AuthCredential(BaseModel):
    auth_type: AuthType
    api_key: Optional[str] = None
    http: Optional[HttpCredentials] = None
    service_account: Optional[Dict[str, str]] = None


class AuthConfig(BaseModel):
    """Configuration for declaring the type of authentication a tool requires."""
    scheme: AuthType
    
    key: str 
    """Identifier used for lookup/storage (e.g., cache / credential store)"""


class ToolContext:
    """
    Context object passed into tool execution.

    Responsibilities:
      - Provide access to runtime state (`state`)
      - Store and retrieve authentication credentials
      - Allow tools to request credentials from external sources (UI/API)
    """

    def __init__(self, function_call_id: str, state: Optional[State] = None):
        self.function_call_id = function_call_id
        self.state = state
        self._credentials: Dict[str, AuthCredential] = {}

    def request_credential(self, auth_config: AuthConfig) -> None:
        """Trigger external workflow (UI/API) to request credentials from the user."""
        # TODO: Implement a real interaction mechanism (e.g. UI message / API callback)
        print(f"[ToolContext] Requesting credential for {auth_config.scheme}, key={auth_config.key}")

    def get_auth(self, auth_config: AuthConfig) -> Optional[AuthCredential]:
        """Retrieve an existing credential if already provided."""
        return self._credentials.get(auth_config.key)

    def set_auth(self, auth_config: AuthConfig, credential: AuthCredential):
        """Store a credential for future use."""
        self._credentials[auth_config.key] = credential

    def require_auth(self, auth_config: AuthConfig) -> AuthCredential:
        """
        Ensure the tool has access to the required credential.
        If missing, trigger credential request.
        """
        cred = self.get_auth(auth_config)
        if cred:
            return cred

        # No credential found, trigger a request
        self.request_credential(auth_config)
        raise RuntimeError(f"Missing required credential: {auth_config.key}")
