// pages/_app.js
import { ChakraProvider } from '@chakra-ui/react';
import { SocketProvider } from '../context/SocketContext';

function MyApp({ Component, pageProps }) {
  return (
    <ChakraProvider>
      <SocketProvider>
        <Component {...pageProps} />
      </SocketProvider>
    </ChakraProvider>
  );
}

export default MyApp;
