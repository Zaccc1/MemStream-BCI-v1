from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loader import connect_one


EIDS = [
    "1a507308-c63a-4e02-8f32-3239a07dc578",
    "4fa70097-8101-4f10-b585-db39429c5ed0",
    "6f6d2c8e-28be-49f4-ae4d-06be2d3148c1",
    "7f150b7c-c261-46e6-9edb-cc391c9d9f03",
    "7f5df7eb-cf36-4589-a20a-14b535441142",
    "83d85891-bd75-4557-91b4-1cbb5f8bfc9d",
    "8c552ddc-813e-4035-81cc-3971b57efe65",
    "9a6e127b-bb07-4be2-92e2-53dd858c2762",
    "9b5a1754-ac99-4d53-97d3-35c2f6638507",
    "a4000c2f-fa75-4b3e-8f06-a7cf599b87ad",
    "b658bc7d-07cd-4203-8a25-7b16b549851b",
    "c16d3557-b2c1-4545-93d0-112ac0915d93",
    "ee8b36de-779f-4dea-901f-e0141c95722b",
    "f304211a-81b1-446f-a435-25e589fe3a5a",
]

FIELDS = (
    "id",
    "subject",
    "start_time",
    "number",
    "lab",
    "projects",
    "task_protocol",
    "procedures",
)


def main() -> None:
    one = connect_one(silent=True)
    records = []
    for eid in EIDS:
        session = one.alyx.rest("sessions", "read", id=eid)
        records.append({field: session.get(field) for field in FIELDS})
        print(
            f"{eid}\t{session.get('subject')}\t{session.get('start_time')}\t"
            f"{session.get('lab')}\t{session.get('projects')}",
            flush=True,
        )
    print("\nJSON\n" + json.dumps(records, indent=2, default=str))

    print("\nRELEASE TAG INTERSECTIONS")
    for tag in ("Brainwidemap", "RepeatedSite"):
        tagged = one.alyx.rest(
            "sessions",
            "list",
            django=f"data_dataset_session_related__tags__name,{tag}",
        )
        tagged_eids = {record["id"] for record in tagged}
        intersection = [eid for eid in EIDS if eid in tagged_eids]
        print(f"{tag}\t{len(intersection)}/{len(EIDS)}\t{intersection}")


if __name__ == "__main__":
    main()
