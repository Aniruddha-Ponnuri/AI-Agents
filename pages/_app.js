import { useState, useEffect } from 'react';
import Head from 'next/head';
import '../styles/globals.css';
import '../styles/layout.css';
import '../styles/chat.css';


function MyApp({ Component, pageProps }) {
  // Add this to prevent hydration errors with SSR
  const [mounted, setMounted] = useState(false);
  
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  return (
    <>
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>CrewAI Data Analysis Platform</title>
      </Head>
      <Component {...pageProps} />
    </>
  );
}

export default MyApp;
