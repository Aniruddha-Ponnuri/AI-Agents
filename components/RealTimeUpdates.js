import { useEffect, useState } from 'react';
import { Box, Text, Badge, List, ListItem, ListIcon, Heading } from '@chakra-ui/react';
import { AiFillCheckCircle } from 'react-icons/ai';
import { useSocket } from '../context/SocketContext';

export default function RealTimeUpdates({ onNewData }) {
  const { socket, isConnected } = useSocket();
  const [updates, setUpdates] = useState([]);

  useEffect(() => {
    if (!socket) return;

    socket.on('update', (data) => {
      const timestamp = new Date().toLocaleTimeString();
      setUpdates(prev => [{ timestamp, data }, ...prev].slice(0, 5));
      
      if (onNewData) {
        onNewData(data);
      }
    });

    return () => {
      socket.off('update');
    };
  }, [socket, onNewData]);

  return (
    <Box p={5} borderWidth="1px" borderRadius="lg">
      <Heading size="md" mb={2}>
        Real-time Updates
        <Badge ml={2} colorScheme={isConnected ? "green" : "red"}>
          {isConnected ? "Connected" : "Disconnected"}
        </Badge>
      </Heading>
      
      {updates.length > 0 ? (
        <List spacing={3}>
          {updates.map((update, index) => (
            <ListItem key={index}>
              <ListIcon as={AiFillCheckCircle} color="green.500" />
              {update.timestamp}: New file processed - {update.data.file_id}
            </ListItem>
          ))}
        </List>
      ) : (
        <Text>No updates yet. Upload a file to see real-time updates.</Text>
      )}
    </Box>
  );
}
