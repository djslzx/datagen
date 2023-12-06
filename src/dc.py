from dataclasses import dataclass
from typing import Any, List, Dict, Set, Optional, Union
from typing import TypedDict


@dataclass
class Result:
    passed: bool
    result: str


@dataclass
class KVItem:
    id: str
    key: str
    value: Any

    @staticmethod
    def from_dict(d: Dict) -> List["KVItem"]:
        id = d["id"]
        return [
            KVItem(id=id, key=key, value=value)
            for key, value in d.items()
            if key != "id"
        ]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "key": self.key,
            "value": self.value,
        }


def test_from_dict():
    cases = [
        {"id": 0,
         "a": 1,
         "b": 2},
        [KVItem(0, "a", 1),
         KVItem(0, "b", 2)],
    ]
    for x, y in zip(cases[::2], cases[1::2]):
        assert KVItem.from_dict(x) == y


def test_json_dump():
    import json
    cases = [
        KVItem(0, "a", 1), {"id": 0, "key": "a", "value": 1},
        KVItem(0, "b", 2), {"id": 0, "key": "b", "value": 2},
    ]
    for x, y in zip(cases[::2], cases[1::2]):
        assert json.dumps(x) == y
