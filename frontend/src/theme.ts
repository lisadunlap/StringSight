import { createTheme } from "@mui/material/styles";

// Professional, neutral base with subtle indigo accents
export const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#4C6EF5", // indigo
    },
    secondary: {
      main: "#7C3AED", // purple accent
    },
    background: {
      default: "#F8FAFC",
      paper: "#FFFFFF",
    },
    text: {
      primary: "#111827",
      secondary: "#6B7280",
    },
    divider: "#E5E7EB",
  },
  shape: { borderRadius: 8 },
  typography: {
    fontFamily: [
      "Inter",
      "ui-sans-serif",
      "system-ui",
      "-apple-system",
      "Segoe UI",
      "Roboto",
      "Helvetica Neue",
      "Arial",
      "Noto Sans",
      "sans-serif",
    ].join(","),
    h6: { fontWeight: 600 },
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: "#FFFFFF",
          color: "#111827",
          borderBottom: "1px solid #E5E7EB",
          boxShadow: "none",
        },
      },
    },
    MuiButton: {
      defaultProps: { disableElevation: true },
      styleOverrides: {
        root: { textTransform: "none", borderRadius: 8 },
      },
    },
    MuiTextField: {
      defaultProps: { size: "small", variant: "outlined" },
    },
    MuiContainer: {
      styleOverrides: {
        root: { paddingTop: 16, paddingBottom: 16 },
      },
    },
  },
});


