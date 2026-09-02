# SSH access to compute clusters — setup guide

Reusable reference for setting up passwordless SSH access from a Claude Code environment to a
remote HPC cluster. Written after setting up DEVANA access from this machine (2026-08-31/09-01).
No secrets are stored here — SSH keys are generated locally and never leave the machine; passwords
(if a cluster requires them at all) are never typed into a command or committed anywhere.

## General method (works for any new cluster)

1. **Check reachability first**, before doing anything else — confirms whether the network path is
   even open:
   ```bash
   getent hosts <hostname>                                   # DNS resolves?
   timeout 5 bash -c 'cat < /dev/null > /dev/tcp/<hostname>/<port>' && echo REACHABLE
   ```

2. **Generate a dedicated keypair** for this cluster — don't reuse a key already tied to another
   host/purpose:
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_<cluster> -N "" -C "<user>@<this-machine>-<cluster>"
   ```
   `-N ""` = no passphrase (convenient for non-interactive/automated use; add a passphrase instead
   if the threat model calls for it — then an `ssh-agent` is needed too).

3. **Get the public key authorized on the remote end.** This is the one step that needs *some*
   existing access — either:
   - an already-working session on another machine that already has key or password access to the
     same account (the DEVANA case: appended via an existing session on "HAL"), or
   - the cluster's own web portal/ticket system if no prior access exists.

   Print the public key and hand it to whichever access path is available:
   ```bash
   cat ~/.ssh/id_ed25519_<cluster>.pub
   ```
   Then, in a session that's already authenticated to the remote account:
   ```bash
   echo "<paste the full ssh-ed25519 AAAA... line>" >> ~/.ssh/authorized_keys
   ```
   Never type the account password into this (or any) session — if password auth is the only path
   available and no other authenticated session exists, that's a case to hand back to the user
   rather than typing a password into a tool call.

4. **Accept the host key** (first-connection step, unrelated to the key pair above):
   ```bash
   ssh-keyscan -p <port> <hostname> >> ~/.ssh/known_hosts
   ```

5. **Add an alias to `~/.ssh/config`** so future sessions (including fresh Claude Code sessions)
   just need `ssh <alias>`:
   ```
   Host <alias>
       HostName <hostname>
       Port <port>
       User <remote-username>
       IdentityFile ~/.ssh/id_ed25519_<cluster>
       IdentitiesOnly yes
   ```

6. **Test non-interactively** (`BatchMode=yes` fails fast instead of hanging on a password prompt
   if key auth isn't actually working yet):
   ```bash
   ssh -o BatchMode=yes -o ConnectTimeout=10 <alias> "hostname && whoami"
   ```

## Known clusters (this project's history — reuse directly if the new project targets the same ones)

### DEVANA (NSCC Slovakia)
- `Host devana` / `login.devana.nscc.sk` / port `5522` / user `adsr1984`
- Key: `~/.ssh/id_ed25519_devana` (no passphrase), added to DEVANA's `~/.ssh/authorized_keys`
  2026-08-31 via an already-working session on the user's other machine ("HAL").
- Storage: project files live under `/project/<project-id>/...` (not `/home/projects/...`).
- Slurm needs `--account=<billing-project>` for GPU jobs — **ask the user which project ID is
  current**, don't assume (multiple exist on this account, meant for different sub-projects/
  allocations, and they get exhausted/rotated — see `cluster_configs/README.md`'s DEVANA section
  for the detailed, evolving history of which project IDs are live).
- Full VASP job-submission conventions (module names, sizing, recipe): `cluster_configs/README.md`
  and `cluster_configs/devana/submit_template_gpu.slurm`.

### FG ("Funglass", nodes ai1/ai2)
- No separate SSH key setup was needed for this project — the working machine ("HAL") turned out
  to already be `cn01`, a node on the same internal cluster network as ai1/ai2, reachable directly
  via `/etc/hosts` entries under the same user account. **If setting this up from a genuinely
  different/external machine, treat it like any other cluster** and follow the general method above.
- No Slurm accounting plugin configured — `--account` is not applicable/needed for FG jobs.
- Full conventions: `cluster_configs/README.md`'s FG section and `cluster_configs/fg/submit_template.slurm`.

## Copying this to another project

This file has no project-specific paths beyond the "Known clusters" section — safe to copy
wholesale into a new project's `cluster_configs/` (or wherever that project keeps cluster docs) and
hand to a fresh Claude Code session there. If the new project targets DEVANA or FG again, the SSH
keys generated here are machine-specific (tied to *this* computer, not the project) — a session
running on a *different* machine will still need to go through the general method once, even if
the target cluster is the same one already documented here.
