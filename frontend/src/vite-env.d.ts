/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_BACKEND?: string
  // add more env variables as needed
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
