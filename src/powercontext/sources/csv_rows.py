# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Row-level CSV Sources with stable identity and canonical text evidence."""

from __future__ import annotations

import csv
import io
import json
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TextEvidence

SOURCE_NAME = "csv-row"
SOURCE_VERSION = "1"
IDENTITY_PREFIX = "csv:"


class CSVDocumentInput(BaseModel):
    """Complete offline CSV bytes plus the exact header column used as key."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One validated CSV record described by its original header and values."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """A captured row-level Source identified by document and primary key."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def _validate_document_id(document_id: object) -> str:
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must be a non-empty string")
    if not document_id.strip():
        raise ValueError("document_id must not be blank")
    if document_id != document_id.strip():
        raise ValueError("document_id must not contain leading or trailing whitespace")
    return document_id


def _validate_primary_key_value(primary_key_value: object) -> str:
    if not isinstance(primary_key_value, str) or not primary_key_value:
        raise ValueError("primary_key_value must be a non-empty string")
    if not primary_key_value.strip():
        raise ValueError("primary_key_value must not be blank")
    return primary_key_value


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the stable, delimiter-safe identity of one CSV row Source."""

    document = _validate_document_id(document_id)
    key_value = _validate_primary_key_value(primary_key_value)
    source_id = f"{IDENTITY_PREFIX}{quote(document, safe='')}:{quote(key_value, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"csv row source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse one whole CSV document into validated row inputs or fail atomically."""

    document_id = _validate_document_id(value.document_id)
    primary_key = value.primary_key
    if not isinstance(primary_key, str) or not primary_key:
        raise ValueError("primary_key must be a non-empty string")

    content = value.content
    if not isinstance(content, bytes):
        raise ValueError("content must be bytes")
    text = content.decode("utf-8-sig")

    try:
        records = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise ValueError(f"malformed CSV content: {error}") from error

    if not records:
        raise ValueError("CSV content must contain a header record")

    header = records[0]
    if not header:
        raise ValueError("CSV header must not be empty")
    for column in header:
        if not column or not column.strip():
            raise ValueError("CSV header columns must be non-empty and non-blank")
    if len(set(header)) != len(header):
        raise ValueError("CSV header columns must be unique")
    if primary_key not in header:
        raise ValueError(f"primary_key {primary_key!r} is not a CSV header column")

    columns = tuple(header)
    key_index = header.index(primary_key)
    seen: set[str] = set()
    rows: list[CSVRowInput] = []
    for record in records[1:]:
        if not record:
            continue
        if len(record) != len(columns):
            raise ValueError("every CSV record must match the header width")
        key_value = record[key_index]
        if not key_value or not key_value.strip():
            raise ValueError("primary key values must be non-empty and non-blank")
        if key_value in seen:
            raise ValueError("primary key values must be unique within a document")
        csv_row_source_id(document_id, key_value)
        seen.add(key_value)
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=key_value,
                columns=columns,
                values=tuple(record),
            )
        )

    return tuple(rows)


class _CSVRowAdapter:
    """Resolve and read row-level CSV Sources without touching a catalog."""

    name = SOURCE_NAME
    input_class = CSVRowInput
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class _CSVRowTextEvidenceProjection:
    """Project one CSV row into the standard ``powercontext.text-evidence`` shape."""

    name = "powercontext.text-evidence"
    version = "1"
    source_class = CSVRowSource
    output_class = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row: dict[str, str] = {}
        for column, field_value in zip(source.columns, source.values, strict=True):
            row[column] = field_value
        metadata: dict[str, Any] = {
            "document_id": source.document_id,
            "primary_key": source.primary_key,
            "primary_key_value": source.primary_key_value,
            "columns": list(source.columns),
            "row": dict(row),
        }
        return TextEvidence(
            source_type=SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata=metadata,
        )


CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = AdapterSourceDefinition(
    adapter=_CSVRowAdapter(),
    version=SOURCE_VERSION,
    projections=(_CSVRowTextEvidenceProjection(),),
)


__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]
