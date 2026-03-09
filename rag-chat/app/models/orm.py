from uuid import uuid4

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, func

from app.db import Base


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String, index=True, nullable=False)
    title = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    model = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    conversation_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=True)
    tool_calls = Column(Text, nullable=True)
    tool_call_id = Column(String, nullable=True)
    seq = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String, index=True, nullable=False)
    name = Column(String, nullable=True)
    prefix = Column(String, nullable=False, index=True)
    hashed_key = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)


class ProviderConfig(Base):
    __tablename__ = "provider_configs"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String, index=True, nullable=False)
    provider_name = Column(String, nullable=False, index=True)
    encrypted_api_key = Column(Text, nullable=True)
    settings_json = Column(Text, nullable=True)
    enabled = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class Generation(Base):
    __tablename__ = "generations"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String, index=True, nullable=False)
    provider_name = Column(String, nullable=False)
    capability = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")
    request_params_json = Column(Text, nullable=True)
    result_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class MediaAsset(Base):
    __tablename__ = "media_assets"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    generation_id = Column(String, index=True, nullable=True)
    user_id = Column(String, index=True, nullable=False)
    media_type = Column(String, nullable=False)
    mime_type = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
