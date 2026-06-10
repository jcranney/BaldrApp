{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: let 
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
  in {

    packages.x86_64-linux.default = pkgs.mkShell rec {
      packages = with pkgs; [
        glib
        libGL
        fontconfig
        libxkbcommon
        wayland
        dbus
        freetype
        uv
        nlohmann_json
      ];
      # QT_DEBUG_PLUGINS = 1;
      shellHook = ''
        uv sync
        source .venv/bin/activate
        alias heimbal-sim="$(pwd)/baldrapp/apps/paranal_simulator/heimbal_simulation_servers.sh"
      '';
      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath packages;
    };

  };
}
