import hashlib
from distill_gym.sandbox.base import SandboxSpec
from distill_gym.sandbox.builders.base import SandboxBuilder
from distill_gym.config.schema import SandboxConfig
from distill_gym.registry.builder_registry import BuilderRegistry


@BuilderRegistry.register("git_repository")
class GitRepositorySandboxBuilder(SandboxBuilder):
    def build(self, config: SandboxConfig) -> SandboxSpec:
        volumes = []
        for v in config.volumes:
            vol = {"target": v.target}
            if v.type == "bind" and v.source:
                vol["source"] = v.source
                vol["type"] = "bind"
            elif v.type == "volume" and v.name:
                vol["source"] = v.name
                vol["type"] = "volume"
            elif v.type == "cache":
                vol["source"] = v.name or f"distill-gym-{v.target.replace('/', '_')}"
                vol["type"] = "volume"
            volumes.append(vol)

        container_name = f"distill-gym-{hashlib.md5(config.repo_url.encode()).hexdigest()[:8]}"

        env = dict(config.env)
        env.setdefault("DEBIAN_FRONTEND", "noninteractive")

        network_mode = config.network.mode.value if config.network.mode else "bridge"
        if network_mode == "proxy_only":
            network_mode = "bridge"

        return SandboxSpec(
            image=config.image,
            container_name=container_name,
            workdir=config.workdir,
            env=env,
            volumes=volumes,
            network_mode=network_mode,
            labels={"distill-gym": "true", "distill-gym-repo": config.repo_url},
            source_type="git_repository",
            source_url=config.repo_url,
            source_ref=config.ref,
        )

    def validate(self, config: SandboxConfig) -> list[str]:
        errors = super().validate(config)
        if not config.repo_url:
            errors.append("sandbox.repo_url is required for git_repository builder")
        return errors
