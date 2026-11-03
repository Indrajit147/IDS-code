"""CICIoMT2024 schema: feature names and fixed label encodings.

Label codes are fixed here (not inferred from data order) so every run,
machine and milestone shares the same integer encoding.
"""
from __future__ import annotations

FEATURES: list[str] = [
    "Header_Length", "Protocol_Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP",
    "ARP", "ICMP", "IGMP", "IPv", "LLC",
    "Tot_sum", "Min", "Max", "AVG", "Std", "Tot_size", "IAT", "Number",
    "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]
RAW_LABEL_COLS = ("Sub_Attack", "Major_Label")

# 6-class headline target (D1). Order follows the proposal.
CATEGORIES: list[str] = ["benign", "ddos", "dos", "recon", "mqtt", "spoofing"]

# 19 fine-grained classes, grouped by category. Values = raw Sub_Attack strings.
ATTACKS19: list[str] = [
    "Benign",
    "DDoS-ICMP", "DDoS-SYN", "DDoS-TCP", "DDoS-UDP",
    "DoS-ICMP", "DoS-SYN", "DoS-TCP", "DoS-UDP",
    "Recon-OS-Scan", "Recon-Ping-Sweep", "Recon-Port-Scan", "Recon-VulScan",
    "MQTT-DDoS-Connect-Flood", "MQTT-DDoS-Publish-Flood", "MQTT-DoS-Connect-Flood",
    "MQTT-DoS-Publish-Flood", "MQTT-Malformed-Data",
    "ARP-Spoofing",
]
MAJOR_TO_CATEGORY = {"Benign": "benign", "DDoS": "ddos", "DoS": "dos",
                     "Recon": "recon", "MQTT": "mqtt", "Spoofing": "spoofing"}

ATTACK19_CODE = {a: i for i, a in enumerate(ATTACKS19)}
CATEGORY_CODE = {c: i for i, c in enumerate(CATEGORIES)}


def attack_category(attack: str) -> str:
    """Category implied by an attack name (used to cross-check Major_Label)."""
    if attack == "Benign":
        return "benign"
    if attack.startswith("MQTT-"):
        return "mqtt"
    if attack == "ARP-Spoofing":
        return "spoofing"
    return {"DDoS": "ddos", "DoS": "dos", "Recon": "recon"}[attack.split("-")[0]]


ATTACK19_TO_CATEGORY_CODE = [CATEGORY_CODE[attack_category(a)] for a in ATTACKS19]


def label_maps() -> dict:
    return {
        "y_attack19": ATTACKS19,
        "y_category": CATEGORIES,
        "y_binary": ["benign", "attack"],
        "attack19_to_category": ATTACK19_TO_CATEGORY_CODE,
    }


# ---- BLE (Zenodo derivative, packet-level; separate feature space) ----
BLE_FEATURES: list[str] = ["packet_length", "inter_packet_gap", "hci_type",
                           "hci_event_code", "l2cap_length", "cumulative_packet_count"]
