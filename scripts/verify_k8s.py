"""
CloudOS-RL Kubernetes Deployment Verification
==============================================
Verifies the Minikube deployment is healthy.

Checks:
  1. Minikube is running (kubectl cluster-info)
  2. Namespace cloudos-rl exists
  3. CRD cloudworkloads.cloudos.ai is registered
  4. RBAC (ServiceAccount, ClusterRole, ClusterRoleBinding) exist
  5. ConfigMap cloudos-config exists
  6. Deployments are available (bridge, api)
  7. Pods are Running
  8. Services exist
  9. Bridge /metrics reachable via port-forward (localhost:9095)
  10. API /health reachable via port-forward (localhost:8001)

Usage:
  python scripts/verify_k8s.py
  python scripts/verify_k8s.py --bridge-port 9095 --api-port 8001
"""

import argparse
import json
import subprocess
import sys
from typing import List, Optional, Tuple

try:
    import requests
except ImportError:
    raise ImportError("pip install requests")


NS = "cloudos-rl"


def _kubectl(*args, check: bool = False) -> Tuple[int, str, str]:
    """Runs a kubectl command and returns (returncode, stdout, stderr)."""
    cmd    = ["kubectl"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _kubectl_json(*args) -> Optional[dict]:
    """Runs kubectl with -o json and returns parsed dict, or None on failure."""
    rc, out, err = _kubectl(*args, "-o", "json")
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
def check_cluster() -> bool:
    print("\n[cluster]  Checking Minikube / cluster ...")
    rc, out, err = _kubectl("cluster-info")
    if rc == 0:
        first_line = out.split("\n")[0]
        print(f"[cluster]  ✅ {first_line}")
        return True
    else:
        print(f"[cluster]  ❌ kubectl cluster-info failed: {err}")
        print(f"[cluster]     Is Minikube running? Run: minikube status")
        return False


def check_namespace() -> bool:
    print(f"\n[namespace]  Checking namespace '{NS}' ...")
    rc, out, _ = _kubectl("get", "namespace", NS)
    if rc == 0:
        print(f"[namespace]  ✅ Namespace '{NS}' exists")
        return True
    else:
        print(f"[namespace]  ❌ Namespace not found — run: kubectl apply -f infrastructure/k8s/namespace.yaml")
        return False


def check_crd() -> bool:
    print(f"\n[crd]  Checking CRD cloudworkloads.cloudos.ai ...")
    rc, out, _ = _kubectl("get", "crd", "cloudworkloads.cloudos.ai")
    if rc == 0:
        print(f"[crd]  ✅ CRD registered")
        return True
    else:
        print(f"[crd]  ❌ CRD not found — run: kubectl apply -f infrastructure/k8s/crd.yaml")
        return False


def check_rbac() -> bool:
    print(f"\n[rbac]  Checking RBAC resources ...")
    ok = True
    checks = [
        ("serviceaccount", "cloudos-operator", f"-n {NS}"),
        ("clusterrole",    "cloudos-operator-role", ""),
        ("clusterrolebinding", "cloudos-operator-binding", ""),
    ]
    for kind, name, ns_flag in checks:
        cmd = ["get", kind, name]
        if ns_flag:
            cmd += ["-n", NS]
        rc, _, _ = _kubectl(*cmd)
        status = "✅" if rc == 0 else "❌"
        print(f"[rbac]  {status}  {kind}/{name}")
        if rc != 0:
            ok = False
    return ok


def check_configmap() -> bool:
    print(f"\n[configmap]  Checking ConfigMap cloudos-config ...")
    data = _kubectl_json("get", "configmap", "cloudos-config", "-n", NS)
    if data:
        keys = list(data.get("data", {}).keys())
        print(f"[configmap]  ✅ Exists — {len(keys)} keys: {', '.join(keys[:5])}{'...' if len(keys) > 5 else ''}")
        return True
    else:
        print(f"[configmap]  ❌ Not found")
        return False


def check_deployments() -> bool:
    print(f"\n[deployments]  Checking deployments ...")
    ok = True
    for name in ["cloudos-bridge", "cloudos-api"]:
        data = _kubectl_json("get", "deployment", name, "-n", NS)
        if not data:
            print(f"[deployments]  ⚠️   {name} — not found (may not be deployed yet)")
            continue
        status     = data.get("status", {})
        desired    = data.get("spec", {}).get("replicas", 1)
        available  = status.get("availableReplicas", 0)
        ready      = status.get("readyReplicas", 0)
        conditions = status.get("conditions", [])
        avail_cond = next((c for c in conditions if c.get("type") == "Available"), {})
        is_avail   = avail_cond.get("status") == "True"

        symbol = "✅" if is_avail else "⚠️ "
        print(f"[deployments]  {symbol}  {name}  desired={desired}  available={available}  ready={ready}")

        if not is_avail:
            reason = avail_cond.get("reason", "")
            msg    = avail_cond.get("message", "")
            print(f"               Reason: {reason} — {msg[:100]}")
            ok = False
    return ok


def check_pods() -> bool:
    print(f"\n[pods]  Checking pod status ...")
    rc, out, _ = _kubectl("get", "pods", "-n", NS,
                          "-o", "custom-columns=NAME:.metadata.name,STATUS:.status.phase,READY:.status.containerStatuses[0].ready,RESTARTS:.status.containerStatuses[0].restartCount",
                          "--no-headers")
    if rc != 0 or not out:
        print(f"[pods]  ⚠️  No pods found in namespace {NS}")
        return False

    all_ok = True
    for line in out.split("\n"):
        parts  = line.split()
        if len(parts) < 2:
            continue
        name   = parts[0]
        phase  = parts[1] if len(parts) > 1 else "?"
        ready  = parts[2] if len(parts) > 2 else "?"
        symbol = "✅" if phase == "Running" and ready == "true" else "⚠️ "
        print(f"[pods]  {symbol}  {name:45s}  {phase:12s}  ready={ready}")
        if phase not in ("Running", "Succeeded"):
            all_ok = False
    return all_ok


def check_services() -> bool:
    print(f"\n[services]  Checking services ...")
    rc, out, _ = _kubectl("get", "services", "-n", NS, "--no-headers")
    if rc != 0 or not out:
        print(f"[services]  ⚠️  No services found")
        return False
    for line in out.split("\n"):
        if line.strip():
            print(f"[services]  ✅  {line}")
    return True


def check_bridge_metrics(host: str, port: int) -> bool:
    print(f"\n[bridge]  Checking bridge metrics at http://{host}:{port}/metrics ...")
    print(f"[bridge]  (requires: kubectl port-forward svc/cloudos-bridge-svc {port}:9090 -n cloudos-rl)")
    try:
        r = requests.get(f"http://{host}:{port}/metrics", timeout=5)
        if r.status_code == 200:
            cloudos_count = len([l for l in r.text.split("\n")
                                 if l.startswith("cloudos_") and not l.startswith("# ")])
            print(f"[bridge]  ✅ Reachable via port-forward — {cloudos_count} cloudos_* metric lines")
            return True
        else:
            print(f"[bridge]  ❌ HTTP {r.status_code}")
            return False
    except requests.ConnectionError:
        print(f"[bridge]  ⚠️  Not reachable (port-forward not running)")
        print(f"[bridge]     Run: kubectl port-forward svc/cloudos-bridge-svc {port}:9090 -n cloudos-rl")
        return False


def check_api_health(host: str, port: int) -> bool:
    print(f"\n[api]  Checking API health at http://{host}:{port}/health ...")
    print(f"[api]  (requires: kubectl port-forward svc/cloudos-api-svc {port}:8000 -n cloudos-rl)")
    try:
        r = requests.get(f"http://{host}:{port}/health", timeout=5)
        if r.status_code == 200:
            print(f"[api]  ✅ API healthy — {r.json()}")
            return True
        else:
            print(f"[api]  ⚠️  HTTP {r.status_code} (API may still be starting)")
            return False
    except requests.ConnectionError:
        print(f"[api]  ⚠️  Not reachable (API not deployed yet — Module C/H completes this)")
        return False


def check_cloudworkload_crd_usable() -> bool:
    """Verifies the CRD is usable by listing CloudWorkload resources."""
    print(f"\n[crd]  Verifying CloudWorkload CRD is queryable ...")
    rc, out, err = _kubectl("get", "cloudworkloads", "-n", NS)
    if rc == 0:
        print(f"[crd]  ✅ kubectl get cloudworkloads works (no resources yet — expected)")
        return True
    else:
        print(f"[crd]  ❌ {err[:120]}")
        return False


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-port", default=9095, type=int)
    parser.add_argument("--api-port",    default=8001, type=int)
    parser.add_argument("--host",        default="localhost")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║    CloudOS-RL  Kubernetes Verification           ║")
    print("╚══════════════════════════════════════════════════╝")

    results = {
        "cluster":    check_cluster(),
        "namespace":  check_namespace(),
        "crd":        check_crd(),
        "rbac":       check_rbac(),
        "configmap":  check_configmap(),
        "deployments":check_deployments(),
        "pods":       check_pods(),
        "services":   check_services(),
        "crd_usable": check_cloudworkload_crd_usable(),
        "bridge":     check_bridge_metrics(args.host, args.bridge_port),
        "api":        check_api_health(args.host, args.api_port),
    }

    print()
    print("=" * 54)
    print("  Summary")
    print("=" * 54)
    for check, ok in results.items():
        symbol = "✅" if ok else ("⚠️ " if check in ("bridge", "api", "deployments", "pods") else "❌")
        print(f"  {symbol}  {check}")
    print("=" * 54)

    core_ok = all(results[k] for k in ["cluster", "namespace", "crd", "rbac", "configmap"])
    if core_ok:
        print()
        rc, ip, _ = _kubectl("get", "nodes", "-o",
                             "jsonpath={.items[0].status.addresses[?(@.type=='InternalIP')].address}")
        minikube_ip = ip or "$(minikube ip)"
        print(f"  API NodePort : http://{minikube_ip}:30800")
        print(f"  Minikube UI  : run 'minikube dashboard'")
        print()


if __name__ == "__main__":
    main()
