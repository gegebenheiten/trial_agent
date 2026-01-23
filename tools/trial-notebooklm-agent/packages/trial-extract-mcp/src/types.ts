export type RemoteConfig = {
  enabled: boolean;
  sshHost?: string;
  sshUser?: string;
  remoteDir?: string;
};

export type RunArgs = {
  nctid: string;
  remote?: RemoteConfig;
  localProfileDir: string;
  localStagingDir: string;
  outputDir: string;
  promptPacks: string[];
};

export type FileList = {
  root: string;
  filePaths: string[];
};
