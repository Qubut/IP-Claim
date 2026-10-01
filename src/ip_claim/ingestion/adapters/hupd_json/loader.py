"""Map one HUPD JSON object onto a Patent value object.

Wire keys stay in this adapter. Blank versus missing strings, YYYYMMDD
dates, and classification code lists are the value object's field types.
Ingress is ``model_validate`` so validators see missing keys as ``None``.
"""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter, ValidationError

from ip_claim.ingestion.models import (
    ApplicationNumber,
    Classification,
    Examiner,
    Inventor,
    Patent,
)
from ip_claim.shared.errors import ConfigurationError

from .claim_parser import parse_claims


def patent_from_hupd_dict(raw: dict[str, Any]) -> Patent:
    """Build a validated :class:`Patent` from one HUPD JSON dict.

    Raises:
        ConfigurationError: when the required ``application_number`` is missing.
    """
    try:
        application_number = TypeAdapter(ApplicationNumber).validate_python(
            raw.get('application_number')
        )
    except ValidationError as exc:
        raise ConfigurationError('HUPD record missing required application_number') from exc

    inventor_list = raw.get('inventor_list')
    inventors = tuple(
        inventor
        for inventor in (
            Inventor.model_validate({
                'last_name': item.get('inventor_name_last'),
                'first_name': item.get('inventor_name_first'),
                'city': item.get('inventor_city'),
                'state': item.get('inventor_state'),
                'country': item.get('inventor_country'),
            })
            for item in (inventor_list if isinstance(inventor_list, list) else ())
            if isinstance(item, dict)
        )
        if inventor.last_name or inventor.first_name
    )
    examiner = Examiner.model_validate({
        'examiner_id': raw.get('examiner_id'),
        'last_name': raw.get('examiner_name_last'),
        'first_name': raw.get('examiner_name_first'),
        'middle_name': raw.get('examiner_name_middle'),
    })
    claims_blob = raw.get('claims')
    return Patent.model_validate({
        'application_number': application_number,
        'publication_number': raw.get('publication_number'),
        'patent_number': raw.get('patent_number'),
        'title': raw.get('title'),
        'decision': raw.get('decision'),
        'filing_date': raw.get('filing_date'),
        'publication_date': raw.get('date_published'),
        'issue_date': raw.get('patent_issue_date'),
        'abandon_date': raw.get('abandon_date'),
        'classification': Classification.model_validate({
            'main_cpc': raw.get('main_cpc_label'),
            'cpc_codes': raw.get('cpc_labels'),
            'main_ipcr': raw.get('main_ipcr_label'),
            'ipcr_codes': raw.get('ipcr_labels'),
            'uspc_class': raw.get('uspc_class'),
            'uspc_subclass': raw.get('uspc_subclass'),
        }),
        'inventors': inventors,
        'examiner': (
            examiner
            if examiner.examiner_id and (examiner.last_name or examiner.first_name)
            else None
        ),
        'abstract': raw.get('abstract'),
        'background': raw.get('background'),
        'summary': raw.get('summary'),
        'full_description': raw.get('full_description'),
        'claims': parse_claims(claims_blob if isinstance(claims_blob, str) else ''),
    })
