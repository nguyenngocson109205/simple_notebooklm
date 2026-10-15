from pydantic import BaseModel, model_validator
from qdrant_client.http import models as qmodels


class MetadataFilter(BaseModel):
    filename: str | None = None
    filenames: list[str] | None = None
    page: int | None = None
    section: str | None = None
    document_id: str | None = None

    @model_validator(mode="after")
    def _normalize(self) -> "MetadataFilter":

        names = [
            n.strip()
            for n in (self.filenames or [])
            if isinstance(n, str) and n.strip()
        ]

        if not names:
            self.filenames = None

        elif len(names) == 1:
            self.filename = names[0]
            self.filenames = None

        else:
            self.filename = None
            self.filenames = names
            self.page = None

        if self.filename is not None:
            self.filename = (
                self.filename.strip()
                or None
            )

        if self.section is not None:
            self.section = (
                self.section.strip()
                or None
            )

        if self.document_id is not None:
            self.document_id = (
                self.document_id.strip()
                or None
            )

        return self


def _coerce_filter(filters):

    if filters is None:
        return None

    if isinstance(
        filters,
        MetadataFilter
    ):
        return filters

    if isinstance(
        filters,
        dict
    ):
        return MetadataFilter(
            **filters
        )

    raise TypeError(
        "filters must be "
        "MetadataFilter, dict, or None"
    )


def filters_to_dict(filters):

    f = _coerce_filter(filters)

    if f is None:
        return None

    data = f.model_dump(
        exclude_none=True
    )

    return data or None


def filters_to_qdrant(filters):

    flat = filters_to_dict(
        filters
    )

    if not flat:
        return None

    conditions = []

    for field, value in flat.items():

        if (
            field == "filenames"
            and isinstance(value, list)
        ):

            conditions.append(
                qmodels.FieldCondition(
                    key="metadata.filename",
                    match=qmodels.MatchAny(
                        any=value
                    )
                )
            )

        elif isinstance(
            value,
            (str, int)
        ):

            conditions.append(
                qmodels.FieldCondition(
                    key=f"metadata.{field}",
                    match=qmodels.MatchValue(
                        value=value
                    )
                )
            )

    if not conditions:
        return None

    return qmodels.Filter(
        must=conditions
    )