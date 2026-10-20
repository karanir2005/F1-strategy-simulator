import { redirect } from "next/navigation";

// the old single-page simulator now lives at /strategy; landing is /predict
export default function Home() {
  redirect("/predict");
}
